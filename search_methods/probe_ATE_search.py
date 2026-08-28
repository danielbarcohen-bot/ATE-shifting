import math
import time
from abc import ABC, abstractmethod
from typing import List, Callable, Set, Tuple, Optional, Dict

import pandas as pd

from search_methods.ATE_search import ATESearch
from utils import apply_data_preparations_seq, calculate_ate_linear_regression_lstsq, \
    get_baseline_ate, df_signature_fast, calculate_ate_with_uncertainty


class ProbManager:
    def __init__(self, operations, columns, op_probs=None, F_elements=None, whole_df_ops=None):
        self.probs = {}
        self.costs = {}
        self.whole_df_ops = whole_df_ops or []
        self._initialize_weights(operations, columns, op_probs, F_elements)

    def _initialize_weights(self, operations: List[str], columns: List[str],
                            op_probs: Optional[Dict[str, float]] = None,
                            F_elements: Optional[List[str]] = None) -> None:

        # Step 1: Determine which elements to consider
        if F_elements is None:
            # Full space: generate all valid combinations
            elements_to_add = self._generate_all_elements(operations, columns)
        else:
            # Restricted space: use only provided elements
            elements_to_add = F_elements

        # Step 2: Assign probabilities and costs to each element
        for f_elem in elements_to_add:
            op, col = self._parse_element(f_elem)

            if op_probs is None:
                # Uniform: each element gets equal probability
                prob = 1.0 / len(elements_to_add)
            else:
                # Weighted: distribute operation's probability among its elements
                op_elements = [elem for elem in elements_to_add if elem.startswith(f"{op}#")]
                num_op_elements = len(op_elements)

                # Each element of this operation gets equal share of op's probability
                prob = op_probs[op] / num_op_elements

            self.probs[f_elem] = prob
            self.costs[f_elem] = self._calculate_cost(f_elem)

    def _generate_all_elements(self, operations: List[str], columns: List[str]) -> List[str]:
        elements = []
        for op in operations:
            if op in self.whole_df_ops:
                elements.append(f"{op}#TABLE")
            else:
                for col in columns:
                    elements.append(f"{op}#{col}")
        return elements

    def _parse_element(self, f_elem: str) -> tuple:
        op, col = f_elem.split("#", 1)  # split on first # only
        return op, col

    # def _initialize_weights(self, operations, columns, op_probs, F_elements=None):
    #     if F_elements is not None:
    #         # Iterate operations in SAME ORDER as else branch
    #         for op in operations:
    #             if op in self.whole_df_ops:
    #                 f_elem = f"{op}#TABLE"
    #                 # Only add if this op#col is in F_elements (already exploded or op is bare whole_df_op)
    #                 if f_elem in F_elements or op in F_elements:
    #                     if op_probs is None:
    #                         prob = 1.0 / len(F_elements)  # Will fix this below
    #                     else:
    #                         prob = op_probs[op] / sum(
    #                             1 for item in F_elements if item.startswith(f"{op}#"))  # len(columns)
    #                     self.probs[f_elem] = prob
    #                     self.costs[f_elem] = self._calculate_cost(f_elem)
    #             else:
    #                 for col in columns:
    #                     f_elem = f"{op}#{col}"
    #                     # Only add if this op#col is in F_elements (already exploded or op is bare whole_df_op)
    #                     if f_elem in F_elements or op in F_elements:
    #                         if op_probs is None:
    #                             prob = 1.0 / len(F_elements)  # Will fix this below
    #                         else:
    #                             prob = op_probs[op] / sum(
    #                                 1 for item in F_elements if item.startswith(f"{op}#"))  # len(columns)
    #                         self.probs[f_elem] = prob
    #                         self.costs[f_elem] = self._calculate_cost(f_elem)
    #     else:
    #         F_size = ((len(operations) - len(self.whole_df_ops)) * len(columns)) + len(self.whole_df_ops)
    #         for op in operations:
    #             if op in self.whole_df_ops:
    #                 print(f"REMEMBER - whole df ops is {self.whole_df_ops}")
    #                 f_elem = f"{op}#TABLE"
    #                 if op_probs is None:
    #                     prob = 1.0 / F_size  # (len(operations) * len(columns))
    #                 else:
    #                     prob = op_probs[op]
    #                 self.probs[f_elem] = prob
    #                 self.costs[f_elem] = self._calculate_cost(f_elem)
    #
    #             else:
    #                 for col in columns:
    #                     f_elem = f"{op}#{col}"
    #                     if op_probs is None:
    #                         prob = 1.0 / (F_size)
    #                     else:
    #                         prob = op_probs[op] / len(columns)
    #                     self.probs[f_elem] = prob
    #                     self.costs[f_elem] = self._calculate_cost(f_elem)

    def _calculate_cost(self, rule_name: str):
        return int(math.ceil(-math.log2(self.probs[rule_name])))

    def get_sequence_probability(self, sequence):
        probability = 1
        for func_name, col in sequence:
            rule_name = f"{func_name}#{col}"
            probability *= self.probs[rule_name]
        return probability

    def update_weights(self, probe_sequence, alpha=0.2):
        rule_counts = {}
        for op, col in probe_sequence:
            rule_name = f"{op}#{col}"
            rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1

        # Create empirical distribution
        empirical = {r: 0.0 for r in self.probs.keys()}
        total = sum(rule_counts.values())
        for rule_name, count in rule_counts.items():
            empirical[rule_name] = count / total

        # Interpolate and renormalize
        new_probs = {}
        for rule_name in self.probs.keys():
            new_probs[rule_name] = (1.0 - alpha) * self.probs[rule_name] + alpha * empirical.get(rule_name, 0.0)

        # Renormalize to ensure sum = 1
        total = sum(new_probs.values())
        self.probs = {k: v / total for k, v in new_probs.items()}

        for rule_name in self.probs.keys():
            self.costs[rule_name] = self._calculate_cost(rule_name)


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

        self.seen_dfs.append(df.copy())
        return True

    def reset(self, df: pd.DataFrame, common_causes: List[str]) -> None:
        self.seen_dfs = [df.copy()]


class ProbeATESearch(ATESearch):
    def __init__(self, use_restart=True, op_probs=None, is_brute=False, use_hash=True):
        self.use_restart = use_restart
        self.op_probs = op_probs
        self.is_brute = is_brute
        self._duplicate_detector = self._create_duplicate_detector(is_brute, use_hash)
        self.transformations_dict = None  # Will be set during search
        self.F_elements = None
        self.whole_df_ops = None

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
               F_elements: List[str] = None, whole_df_ops: List[str] = None):
        # Store for access in helper methods
        self.transformations_dict = transformations_dict
        self.F_elements = F_elements
        self.whole_df_ops = whole_df_ops
        # Precompute function prefixes once at startup
        func_prefixes = {func_name: func_name.split("_")[0] for func_name in transformations_dict.keys()}
        df_ = df.copy()
        baseline_ate = get_baseline_ate(common_causes, df_)
        print(f"START ATE IS: {baseline_ate}")

        bank = {0: [()]}  # init with the empty sequence
        self._duplicate_detector.reset(df_, common_causes)

        prob_manager = ProbManager([func_name for func_name, func in transformations_dict.items()],
                                   common_causes, self.op_probs, F_elements, whole_df_ops)
        cost = 1
        best_ate_error = float('inf')

        smallest_distance_from_target = abs(baseline_ate - target_ate)
        distances_at_time_from_target = [(smallest_distance_from_target, 0)]
        checked = 0
        start_time = time.time()

        while True:
            if time.time() - start_time > time_out_sec:
                print("\n\n*** TIMED OUT!! ***\n")
                print(f"distances from ATE (with time):\n{distances_at_time_from_target}", flush=True)
                break

            should_restart = False
            bank[cost] = []

            for move in self.moves_under_cost(cost, prob_manager):
                if should_restart:
                    break

                for seq in bank[cost - prob_manager.costs[move]]:
                    func_name, col = move.split("#")
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
                    new_ate = calculate_ate_linear_regression_lstsq(curr_df, 'treatment', 'outcome', common_causes)
                    current_error = abs(new_ate - target_ate)

                    # Track progress
                    if current_error < smallest_distance_from_target:
                        smallest_distance_from_target = current_error
                        distances_at_time_from_target.append((current_error, time.time() - start_time))

                    # Found solution within tolerance
                    if current_error < epsilon:
                        self._print_solution(new_seq, baseline_ate, new_ate, prob_manager,
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
                        smallest_distance_from_target = abs(baseline_ate - target_ate)
                        distances_at_time_from_target.append((smallest_distance_from_target, time.time() - start_time))

                        bank.clear()
                        bank[0] = [()]
                        self._duplicate_detector.reset(df_, common_causes)

                        cost = 1
                        should_restart = True
                        best_ate_error = current_error

                    if should_restart:
                        break

            if not should_restart:
                cost += 1

    def _print_solution(self, solution_seq: Tuple, baseline_ate: float, new_ate: float,
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
                [func_name for func_name, func in self.transformations_dict.items()],
                common_causes, self.op_probs, self.F_elements, self.whole_df_ops)
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
        return [move for move in prob_manager.probs.keys()
                if prob_manager.costs[move] <= cost]
