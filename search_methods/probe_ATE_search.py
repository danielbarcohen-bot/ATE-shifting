import math
import time
from typing import List, Callable

import pandas as pd

from search_methods.ATE_search import ATESearch
from utils import apply_data_preparations_seq, calculate_ate_linear_regression_lstsq, \
    get_base_line, df_signature_fast


class ProbManager:
    def __init__(self, operations, columns):
        self.probs = {}
        self.costs = {}
        self._initialize_weights(operations, columns)

    def _initialize_weights(self, operations, columns):
        for op in operations:
            for col in columns:
                rule_name = f"{op}#{col}"
                prob = 1.0 / (len(operations) * len(columns))  # Uniform initial weight
                self.probs[rule_name] = prob
                self.costs[rule_name] = self._get_cost(rule_name)

    def _get_cost(self, rule_name: str):
        return int(math.ceil(-math.log2(self.probs[rule_name])))

    def update_weights(self, probe_sequence, alpha=0.2):

        """
        Updates weights using the interpolation method based on a successful probe.
        """
        # Calculate the empirical distribution D_probe from the sequence
        rule_counts = {}
        for op, col in probe_sequence:
            rule_name = f"{op}#{col}"
            rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1

        # 1. Update Operation Selection Rules ('O')
        for rule_name in self.probs.keys():
            # D_probe is 1 if the rule was used, 0 otherwise (in this simplified view)
            is_used = 1.0 if rule_name in rule_counts else 0.0

            # Interpolation: W_new = (1-a)*W_current + a*W_probe
            self.probs[rule_name] = (1.0 - alpha) * self.probs[rule_name] + alpha * is_used
            self.costs[rule_name] = self._get_cost(rule_name)


class ProbeATESearch(ATESearch):

    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
               max_seq_length: int, transformations_dict: dict[str, Callable]):
        df_ = df.copy()
        base_line_ate = get_base_line(common_causes, df_)
        print(f"START ATE IS: {base_line_ate}")
        bank = {0: [()]}  # init with the empty sequence
        seen_dfs = {df_signature_fast(df.copy(), common_causes)}
        prob_manager = ProbManager([func_name for func_name, func in transformations_dict.items()], common_causes)
        cost = 1
        best_ate_error = float('inf')

        smallest_distance_from_target = abs(base_line_ate - target_ate)
        distances_at_time_from_target = [(smallest_distance_from_target, 0)]
        checked = 0
        start_time = time.time()
        while True:
            should_restart = False
            bank[cost] = []
            for move in self.moves_under_cost(cost, prob_manager):
                if should_restart:
                    break
                for seq in bank[cost - prob_manager.costs[move]]:
                    func_name, col = move.split("#")
                    new_seq = seq + ((func_name, col),)

                    if len(new_seq) > max_seq_length:
                        continue

                    if any(f_n.split("_")[0] == func_name.split("_")[0] for f_n, c in seq if c == col):
                        continue
                    checked = checked + 1
                    curr_df = apply_data_preparations_seq(df_, new_seq, transformations_dict)
                    new_ate = calculate_ate_linear_regression_lstsq(curr_df, 'treatment', 'outcome',
                                                                    common_causes)
                    current_error = abs(new_ate - target_ate)

                    if current_error < smallest_distance_from_target:
                        smallest_distance_from_target = current_error
                        distances_at_time_from_target.append((current_error, time.time() - start_time))

                    if current_error < epsilon:
                        print(
                            f"""***\nFINISHED\nATE before: {base_line_ate}\nATE now is: {new_ate}\nsequence is: {new_seq}\n***""",
                            flush=True)
                        solution_seq = new_seq
                        print(f"Execution time: {time.time() - start_time:.3f} sec")
                        print(f"distances from ATE (with time):\n{distances_at_time_from_target}", flush=True)
                        print(f"checked:\n{checked}", flush=True)

                        exit()
                    df_new_signature = df_signature_fast(curr_df, common_causes)

                    if df_new_signature in seen_dfs:
                        break
                    bank[cost].append(new_seq)
                    seen_dfs.add(df_new_signature)
                    if current_error < best_ate_error * 0.9:
                        print(
                            f"PROBE TRIGGERED! Error reduced from {best_ate_error:.3f} to {current_error:.3f} (ATE went to {new_ate}).")
                        prob_manager.update_weights(new_seq)

                        smallest_distance_from_target = abs(base_line_ate - target_ate)
                        distances_at_time_from_target.append((smallest_distance_from_target, time.time() - start_time))

                        best_ate_error = current_error  # Update the best error seen
                        bank = {0: [()]}
                        cost = 1
                        should_restart = True
                        seen_dfs = {df_signature_fast(df.copy(), common_causes)}

                    if should_restart:
                        break

            if not should_restart:
                cost += 1

    def moves_under_cost(self, cost: int, prob_manager: ProbManager):
        moves = []
        for move in prob_manager.probs.keys():
            if prob_manager.costs[move] <= cost:
                moves.append(move)
        return moves
