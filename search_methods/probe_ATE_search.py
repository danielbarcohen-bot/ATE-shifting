import math
import time
from abc import ABC, abstractmethod
from typing import List, Callable, Set, Tuple, Optional, Dict

import pandas as pd

from utils import apply_data_preparations_seq, calculate_ate_linear_regression_lstsq, \
    df_signature_fast, calculate_ate_with_uncertainty, get_fill_combinations, get_fill_options


class ProbManager:
    def __init__(self,
                 operations,  # not fill just ops, format [(op,col)]
                 columns,
                 op_probs=None,
                 F_elements=None,
                 whole_df_ops=None,
                 legal_ops_by_type=None,
                 legal_fill_by_type=None,
                 col_types=None, fill_columns=None):
        self.probs = {}
        self.costs = {}
        self.whole_df_ops = whole_df_ops or []
        self.col_types = col_types or {}
        self.legal_ops_by_type = legal_ops_by_type or {}
        self.legal_fill_by_type = legal_fill_by_type or {}
        fill_ops = get_fill_options(fill_columns, col_types, self.legal_fill_by_type)
        operations = [(op, col) for op in operations for col in columns if op in legal_ops_by_type[col_types[col]]]
        operations += [(op, 'TABLE') for op in whole_df_ops]
        if F_elements is not None:
            operations = list(filter(lambda op_col: op_col[0] + "#" + op_col[1] in F_elements, operations))
        self.fill_columns = fill_columns or []
        self._initialize_weights(
            operations,
            fill_ops,
            columns,
            op_probs
        )

    def get_sequence_cost(self, sequence) -> int:
        return sum(self.costs[(func_name, col)] for func_name, col in sequence)

    def _initialize_weights(self,
                            operations: List[Tuple[str, str]],
                            fill_ops: List[Tuple[str, str]],
                            columns: List[str],
                            op_probs: Optional[Dict[str, float]] = None
                            ) -> None:

        elements_to_add = operations + fill_ops

        # Step 2: Assign probabilities and costs to each element
        for op, col in elements_to_add:
            if op_probs is None:
                # Uniform: each element gets equal probability
                prob = 1.0 / len(elements_to_add)
            else:
                # Weighted: distribute operation's probability among its elements
                op_elements = [elem for elem in elements_to_add if elem[0] == op]
                num_op_elements = len(op_elements)

                # Each element of this operation gets equal share of op's probability
                prob = op_probs[op] / num_op_elements

            self.probs[(op, col)] = prob

        # Step 3: Normalize so all elements (fills included) sum to 1, then derive costs
        total = sum(self.probs.values())
        self.probs = {k: v / total for k, v in self.probs.items()}
        for f_elem in self.probs:
            self.costs[f_elem] = self._calculate_cost(f_elem)

    def _parse_element(self, f_elem: str) -> tuple:
        op, col = f_elem.split("#", 1)  # split on first # only
        return op, col

    def _calculate_cost(self, rule_name: str):
        return int(math.ceil(-math.log2(self.probs[rule_name])))

    def get_sequence_probability(self, sequence):
        probability = 1
        for func_name, col in sequence:
            probability *= self.probs[(func_name, col)]
        return probability

    def update_weights(self, probe_sequence, alpha=0.2):
        rule_counts = {}
        for op, col in probe_sequence:
            rule_counts[(op, col)] = rule_counts.get((op, col), 0) + 1

        empirical = {r: 0.0 for r in self.probs.keys()}
        total = sum(rule_counts.values())
        for rule_key, count in rule_counts.items():
            empirical[rule_key] = count / total

        new_probs = {}
        for rule_key in self.probs.keys():
            new_probs[rule_key] = (1.0 - alpha) * self.probs[rule_key] + alpha * empirical.get(rule_key, 0.0)

        total = sum(new_probs.values())
        self.probs = {k: v / total for k, v in new_probs.items()}

        for rule_key in self.probs.keys():
            self.costs[rule_key] = self._calculate_cost(rule_key)


class DuplicateDetector(ABC):
    """Strategy for detecting duplicate dataframes during search."""

    @abstractmethod
    def add_if_new(self, df: pd.DataFrame, common_causes: List[str]) -> bool:
        """
        Check if df was seen. If NOT seen, register it and return True.
        If already seen, return False.
        """
        pass

    @abstractmethod
    def reset(self, df: pd.DataFrame, common_causes: List[str]) -> None:
        """Clear seen states and reinitialize."""
        pass


class NullDuplicateDetector(DuplicateDetector):
    """Brute-force: always considers states as new (never stores them)."""

    def add_if_new(self, df: pd.DataFrame, common_causes: List[str]) -> bool:
        return True

    def reset(self, df: pd.DataFrame, common_causes: List[str]) -> None:
        pass


class HashDuplicateDetector(DuplicateDetector):
    """Use fast hash signatures to detect duplicates."""

    def __init__(self):
        self.seen_dfs: Set[str] = set()

    def add_if_new(self, df: pd.DataFrame, common_causes: List[str]) -> bool:
        # Signature is calculated ONLY ONCE per dataframe
        signature = df_signature_fast(df, common_causes)
        if signature in self.seen_dfs:
            return False

        self.seen_dfs.add(signature)
        return True

    def reset(self, df: pd.DataFrame, common_causes: List[str]) -> None:
        self.seen_dfs = {df_signature_fast(df, common_causes)}


class EqualityDuplicateDetector(DuplicateDetector):
    """Use dataframe equality checks to detect duplicates."""

    def __init__(self):
        self.seen_dfs: List[pd.DataFrame] = []

    def add_if_new(self, df: pd.DataFrame, common_causes: List[str]) -> bool:
        if any(df.equals(seen) for seen in self.seen_dfs):
            return False

        self.seen_dfs.append(df)
        return True

    def reset(self, df: pd.DataFrame, common_causes: List[str]) -> None:
        self.seen_dfs = [df]


class ProbeATESearch:
    def __init__(self, use_restart=True, op_probs=None, is_brute=False, use_hash=True,
                 seq_ates_writer=None):
        self.use_restart = use_restart
        self.op_probs = op_probs
        self.is_brute = is_brute
        self._duplicate_detector = self._create_duplicate_detector(is_brute, use_hash)
        self.transformations_dict = None  # Will be set during search
        self.F_elements = None
        self.whole_df_ops = None
        self.fill_by_type = None
        self.fill_columns = None
        self.seq_ates_writer = seq_ates_writer

    def _create_duplicate_detector(self, is_brute: bool, use_hash: bool) -> DuplicateDetector:
        """Factory method to create the appropriate duplicate detector."""
        if is_brute:
            return NullDuplicateDetector()
        elif use_hash:
            return HashDuplicateDetector()
        else:
            return EqualityDuplicateDetector()

    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
               transformations_dict: dict[str, Callable], time_out_sec: int = 14400,
               F_elements: List[str] = None, whole_df_ops: List[str] = None, legal_ops_by_type=None,
               fill_by_type=None):
        # Store for access in helper methods
        self.transformations_dict = transformations_dict
        self.F_elements = F_elements
        self.whole_df_ops = whole_df_ops
        self.legal_ops_by_type = legal_ops_by_type
        self.fill_by_type = fill_by_type
        self.col_types = df.attrs.get('col_types', None)
        self.fill_columns = [col for col in df.columns if df[col].isna().any()]
        # Precompute function prefixes once at startup
        func_prefixes = {func_name: func_name.split("_")[0] for func_name in transformations_dict.keys()}
        df_ = df.copy()

        prob_manager = ProbManager(list(transformations_dict.keys()),
                                   common_causes, self.op_probs, F_elements, whole_df_ops, self.legal_ops_by_type,
                                   legal_fill_by_type=self.fill_by_type, col_types=self.col_types,
                                   fill_columns=self.fill_columns)

        # baseline_ate = calculate_ate_linear_regression_lstsq(df_, 'treatment', 'outcome', common_causes)

        if self.fill_columns:
            print(f'No start ate, there are {len(self.fill_columns)} columns with missing values')
        else:
            print(f"START ATE IS: {calculate_ate_linear_regression_lstsq(df_, 'treatment', 'outcome', common_causes)}")

        bank, init_distance, best_init = self._init_bank(df_, common_causes, target_ate, prob_manager)
        if init_distance < epsilon:
            print(f"FOUND SOLUTION WITH NO NEED OF DATA PREP\nsequence is: {best_init}", flush=True)
            return best_init

        cost = 1
        best_ate_error = init_distance

        smallest_distance_from_target = init_distance
        distances_at_time_from_target = [(smallest_distance_from_target, 0)]
        checked = 0
        start_time = time.time()

        while True:
            if time.time() - start_time > time_out_sec:
                print("\n\n*** TIMED OUT!! ***\n")
                print(f"distances from ATE (with time):\n{distances_at_time_from_target}", flush=True)
                break

            should_restart = False
            if cost not in bank:
                bank[cost] = []

            for move in self.moves_under_cost(cost, prob_manager):
                if should_restart:
                    break

                if (cost - prob_manager.costs[move]) not in bank:
                    continue

                for seq in bank[cost - prob_manager.costs[move]]:
                    func_name, col = move
                    new_seq = seq + ((func_name, col),)

                    # Enforce operation repetition rules
                    if whole_df_ops and func_name in whole_df_ops:
                        if any(f_n == func_name for f_n, c in seq):
                            continue
                    curr_prefix = func_prefixes[func_name]
                    if any(func_prefixes[f_n] == curr_prefix for f_n, c in seq if c == col):
                        continue

                    checked += 1
                    curr_df = apply_data_preparations_seq(df_, new_seq, transformations_dict)

                    # Doesn't happen, shame to waste time on even checking an assert.
                    # if curr_df.isna().any().any():
                    #     # if not self._duplicate_detector.add_if_new(curr_df, common_causes):
                    #     #     continue
                    #     # bank[cost].append(new_seq)
                    #     continue

                    new_ate = calculate_ate_linear_regression_lstsq(curr_df, 'treatment', 'outcome', common_causes)
                    current_error = abs(new_ate - target_ate)

                    if self.seq_ates_writer is not None:
                        self.seq_ates_writer.writerow((new_seq, new_ate))

                    # Track progress
                    if current_error < smallest_distance_from_target:
                        smallest_distance_from_target = current_error
                        distances_at_time_from_target.append((current_error, time.time() - start_time))

                    # Found solution within tolerance
                    if current_error < epsilon:
                        self._print_solution(new_seq,
                                             calculate_ate_linear_regression_lstsq(df_, 'treatment', 'outcome',
                                                                                   common_causes) if not self.fill_columns else 'N/A',
                                             new_ate, prob_manager,
                                             curr_df, common_causes, checked, start_time,
                                             distances_at_time_from_target)
                        return new_seq

                    # Skip if we've seen this state before
                    if not self._duplicate_detector.add_if_new(curr_df, common_causes):
                        continue

                    bank[cost].append(new_seq)

                    # Probe trigger: found significant improvement
                    if self.use_restart and current_error < best_ate_error * 0.9:
                        print(
                            f"PROBE TRIGGERED! Error reduced from {best_ate_error:.3f} to {current_error:.3f} (ATE went to {new_ate}).")
                        prob_manager.update_weights(new_seq)

                        # Reset for restart
                        bank, smallest_distance_from_target, _ = self._init_bank(df_, common_causes, target_ate,
                                                                                 prob_manager)
                        distances_at_time_from_target.append((smallest_distance_from_target, time.time() - start_time))

                        cost = 1
                        should_restart = True
                        best_ate_error = current_error

                    if should_restart:
                        break

            if not should_restart:
                cost += 1

    def _init_bank(self, df_: pd.DataFrame, common_causes: List[str], target_ate: float,
                   prob_manager: 'ProbManager') -> Tuple[Dict[int, List[Tuple]], float, Tuple]:
        """Reset the duplicate detector and build the initial bank.
        Returns (bank, smallest distance from target among the initial sequences, best initial sequence)."""
        self._duplicate_detector.reset(df_, common_causes)

        if self.fill_by_type is None or not df_.isnull().values.any():
            baseline_ate = calculate_ate_linear_regression_lstsq(df_, 'treatment', 'outcome', common_causes)
            return {0: [()]}, abs(baseline_ate - target_ate), ()  # init with the empty sequence

        fill_methods = [(seq, prob_manager.get_sequence_cost(seq)) for seq in
                        get_fill_combinations(df_, self.col_types, self.fill_by_type)]
        fill_methods.sort(key=lambda x: x[1])
        print(f"{len(fill_methods)} options to fill missing")
        bank = {}
        smallest_distance_from_target = None
        best_init = None
        for fill_sequence, fill_cost in fill_methods:
            filled_df = apply_data_preparations_seq(df_, fill_sequence, self.transformations_dict)
            if not self._duplicate_detector.add_if_new(filled_df, common_causes):  # OE on the fill sequences
                continue
            new_ate = calculate_ate_linear_regression_lstsq(filled_df, 'treatment', 'outcome', common_causes)
            if fill_cost not in bank:
                bank[fill_cost] = []
            bank[fill_cost].append(fill_sequence)
            if smallest_distance_from_target is None or abs(new_ate - target_ate) < smallest_distance_from_target:
                smallest_distance_from_target = abs(new_ate - target_ate)
                best_init = fill_sequence
        return bank, smallest_distance_from_target, best_init

    def _print_solution(self, solution_seq: Tuple, baseline_ate: float | str, new_ate: float,
                        prob_manager: 'ProbManager', curr_df: pd.DataFrame, common_causes: List[str],
                        checked: int, start_time: float, distances_at_time_from_target: List) -> None:
        """Print the solution details."""
        print(f"""***
FINISHED
ATE before: {baseline_ate}
ATE now is: {new_ate}
sequence is: {solution_seq}
***""", flush=True)

        print(f"Execution time: {time.time() - start_time:.3f} sec")
        print(f"distances from ATE (with time):\n{distances_at_time_from_target}", flush=True)

        if self.use_restart:
            temp_prob_manager = ProbManager(
                list(self.transformations_dict.keys()),
                common_causes, self.op_probs, self.F_elements, self.whole_df_ops, self.legal_ops_by_type,
                legal_fill_by_type=self.fill_by_type, col_types=self.col_types, fill_columns=self.fill_columns)
            print(
                f"(REAL, NOT adjusted by restarts) probability of this sequence is: {temp_prob_manager.get_sequence_probability(solution_seq)}")
        else:
            print(f"probability of this sequence is: {prob_manager.get_sequence_probability(solution_seq)}")

        try:
            print(
                f"uncertainty:\n{calculate_ate_with_uncertainty(curr_df.copy(), 'treatment', 'outcome', common_causes)}")
        except Exception as e:
            print(f"Failed to calculate uncertainty:\n{e}")

        print(f"checked:\n{checked}", flush=True)

    def moves_under_cost(self, cost: int, prob_manager: 'ProbManager') -> List[str]:
        return [f for (f,fcost) in prob_manager.costs.items() if fcost <= cost and not f[0].startswith('fill_')]
