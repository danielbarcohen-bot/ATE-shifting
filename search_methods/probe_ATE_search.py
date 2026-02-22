# import heapq
# import time
# from typing import List, Callable
#
# import numpy as np
# import pandas as pd
#
# from search_methods.ATE_search import ATESearch
# from search_methods.pruning_ATE_search import canonical
# from utils import apply_data_preparations_seq, calculate_ate_linear_regression_lstsq, \
#     df_signature_fast, get_moves_and_moveBit, manual_dml_ate, manual_dr_ate, get_base_line
#
#
# # --- PCFG Class to Manage Weights ---
#
# class PCFG:
#     def __init__(self, operations, columns):
#         self.rules = {}  # Stores {rule_name: probability}
#         self.non_terminals = {}  # Stores {non_terminal: [rule_names]}
#         self._initialize_weights(operations, columns)
#
#     def _initialize_weights(self, operations, columns):
#         # 1. Operation Selection Rules (Non-terminal O)
#         op_rules = []
#         for op in operations:
#             for col in columns:
#                 rule_name = f"{op}_{col}"
#                 self.rules[rule_name] = 1.0 / (len(operations) * len(columns))  # Uniform initial weight
#                 op_rules.append(rule_name)
#         self.non_terminals['O'] = op_rules
#
#         # 2. Sequence Continuation/Termination Rules (Non-terminal S)
#         # S -> O S (Continue)
#         self.rules['S_continue'] = 0.8
#         # S -> E (End)
#         self.rules['S_end'] = 0.2
#         self.non_terminals['S'] = ['S_continue', 'S_end']
#
#     def get_prob(self, rule_name):
#         """Get the probability of a specific rule."""
#         return self.rules.get(rule_name, 0.0)
#
#     def get_cost(self, sequence):
#         """Calculates -log(Probability) for the entire sequence."""
#         if not sequence:
#             return -np.log(self.rules['S_end'])  # Cost of just terminating
#
#         # Start with the probability of the first operation
#         log_prob = 0.0
#
#         # Add probability of each operation choice and sequence continuation
#         for op, col in sequence:
#             # P(O) for the specific operation choice
#             rule_name = f"{op}_{col}"
#             log_prob += np.log(self.rules.get(rule_name, 1e-10))
#             # P(S -> O S) for sequence continuation (except the last one)
#             log_prob += np.log(self.rules.get('S_continue', 1e-10))
#
#         # Add probability of termination
#         log_prob += np.log(self.rules.get('S_end', 1e-10))
#
#         return -log_prob / (len(sequence) + 1)  # -log_prob  # Cost = -log(P)
#
#     def update_weights(self, probe_sequence, alpha=0.2):
#         """
#         Updates weights using the interpolation method based on a successful probe.
#         """
#         # Calculate the empirical distribution D_probe from the sequence
#         rule_counts = {}
#         for op, col in probe_sequence:
#             rule_name = f"{op}_{col}"
#             rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
#
#         # 1. Update Operation Selection Rules ('O')
#         for rule_name in self.non_terminals['O']:
#             # D_probe is 1 if the rule was used, 0 otherwise (in this simplified view)
#             is_used = 1.0 if rule_name in rule_counts else 0.0
#
#             # Interpolation: W_new = (1-a)*W_current + a*W_probe
#             self.rules[rule_name] = (1.0 - alpha) * self.rules[rule_name] + alpha * is_used
#
#         # 2. Normalize the Operation Selection Rules (must sum to 1)
#         # This is a critical step: ensure the weights of competing rules still sum to 1.
#         total_prob = sum(self.rules[r] for r in self.non_terminals['O'])
#         for r in self.non_terminals['O']:
#             self.rules[r] /= total_prob
#
#         # Optional: You could update S_continue/S_end if you want to learn sequence length,
#         # but we keep them fixed here for simplicity.
#
#
# class ProbeATESearch(ATESearch):
#     def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
#                max_seq_length: int, transformations_dict: dict[str, Callable]):
#         """
#         Implements the adaptive Best-First Search guided by the PCFG cost.
#         """
#         pcfg = PCFG([func_name for func_name, func in transformations_dict.items()], common_causes)
#         fast_moves = get_moves_and_moveBit(common_causes, transformations_dict.keys())
#         # Priority Queue stores tuples: (cost, sequence)
#         # The cost is the PCFG cost, NOT the ATE error.
#         pq = []
#         # 1. Start with the empty sequence (Cost calculated from PCFG)
#         initial_sequence = ()
#         initial_cost = pcfg.get_cost(initial_sequence)
#         heapq.heappush(pq, (initial_cost, initial_sequence, 0))
#
#         steps = 0
#         best_ate_error = float('inf')
#
#         seen_seq = set()
#         seen_dfs = set()
#         seen_dfs.add(df_signature_fast(df.copy(), common_causes))
#
#         start_time = time.time()
#         base_line_ate = get_base_line(common_causes, df.copy())
#         # smallest_ate = np.inf
#         # largest_ate = -np.inf
#         # largest_ate_and_time_list = []
#         # smallest_ate_and_time_list = []
#         smallest_distance_from_target = abs(base_line_ate - target_ate)
#         distances_at_time_from_target = [(smallest_distance_from_target, 0)]
#
#         while pq:
#             cost, sequence, mask = heapq.heappop(pq)
#             steps += 1
#
#             # --- ATE Evaluation & Goal Check ---
#             curr_df = apply_data_preparations_seq(df, sequence, transformations_dict)
#             curr_df_filled = curr_df#.dropna()  # curr_df.fillna(value=curr_df.mean())
#             current_ate = calculate_ate_linear_regression_lstsq(curr_df_filled, 'treatment', 'outcome', common_causes)
#             current_error = abs(current_ate - target_ate)
#
#             if current_error < smallest_distance_from_target:
#                 smallest_distance_from_target = current_error
#                 distances_at_time_from_target.append((current_error, time.time() - start_time))
#             # if current_ate > largest_ate:
#             #     largest_ate = current_ate
#             #     #print(f"LARGEST ATE now is {largest_ate}\nit took {time.time() - start_time} sec to get here\nwith sequence:\n{sequence}\n")
#             #     largest_ate_and_time_list.append((largest_ate.item(), time.time() - start_time))
#             #     print(f"smallest lists: {smallest_ate_and_time_list}",flush=True)
#             #     print(f"largest lists: {largest_ate_and_time_list}", flush=True)
#             # if current_ate < smallest_ate:
#             #     smallest_ate = current_ate
#             #     # print(f#"SMALLEST ATE now is {smallest_ate} with sequence:\n{sequence}\n")
#             #     smallest_ate_and_time_list.append((smallest_ate.item(), time.time() - start_time))
#             #
#             #     print(f"smallest lists: {smallest_ate_and_time_list}",flush=True)
#             #     print(f"largest lists: {largest_ate_and_time_list}", flush=True)
#
#
#             if current_error < epsilon:
#                 print(f"\n **GOAL REACHED!** Final Sequence: {sequence} with ATE {current_ate:.7f}")
#                 print(f"run took {time.time() - start_time:.2f} seconds")
#                 print(f"checked {steps} combinations")
#                 print(f"distances from ATE (with time):\n{distances_at_time_from_target}", flush=True)
#                 return sequence, time.time() - start_time
#
#             # --- Probe Trigger (JIT Learning) ---
#             # A probe is triggered if this sequence is significantly better than the best error seen so far.
#             # Here we use a simple check: if the error is reduced by 50%
#             if current_error < best_ate_error * 0.9:  # * 0.5: #TODO: should break here?
#                 print(
#                     f"🔥 PROBE TRIGGERED! Error reduced from {best_ate_error:.3f} to {current_error:.3f} (ATE went to {current_ate}).")
#                 pcfg.update_weights(sequence)
#                 best_ate_error = current_error  # Update the best error seen
#                 # reset the pq:
#                 if pq:
#                     print("💡 Re-scoring search frontier...")
#                     # restart the smallest \ largest ive seen
#                     # smallest_ate = np.inf
#                     # largest_ate = -np.inf
#                     smallest_distance_from_target = abs(base_line_ate - target_ate)
#                     distances_at_time_from_target.append((smallest_distance_from_target, time.time() - start_time))
#
#                     # 1. Extract all items from the current queue
#                     current_frontier = []
#                     while pq:
#                         # We only stored (cost, sequence). We need to extract the sequence.
#                         old_cost, seq, mask_ = heapq.heappop(pq)
#                         current_frontier.append((seq, mask_))
#
#                     # 2. Re-calculate the cost for every sequence using the updated PCFG
#                     for seq, mask_ in current_frontier:
#                         new_cost = pcfg.get_cost(seq)
#                         # 3. Push the sequence back with the new, correct cost
#                         heapq.heappush(pq, (new_cost, seq, mask_))
#
#                     print(f"✅ Frontier refreshed. Queue size: {len(pq)}")
#
#             # --- Expansion (Generating Children) ---
#             # Generate new sequences by appending one operation
#             if len(sequence) < max_seq_length:
#                 for func_name, col, move_bit in fast_moves:
#                     new_mask = mask | move_bit
#                     new_sequence = sequence + ((func_name, col),)
#
#                     # Q.append((new_path, new_mask))
#
#                     if mask & move_bit or (func_name.startswith("fill_") and curr_df[col].isna().sum() == 0):
#                         continue
#                     new_col = transformations_dict[func_name](curr_df[col].copy())
#
#                     canonical_sequence = canonical(new_sequence)
#                     if canonical_sequence in seen_seq or new_col.equals(
#                             curr_df[col]):  # col hasn't changed - so same df!
#                         continue
#
#                     seen_seq.add(canonical_sequence)
#                     new_df = curr_df.copy(deep=False)
#                     new_df[col] = new_col
#                     df_new_signature = df_signature_fast(new_df, common_causes)
#
#                     if df_new_signature in seen_dfs:
#                         continue
#
#                     # if df hasnt been explored:
#                     else:
#                         # print(f"added func {func_name}", flush=True)
#                         seen_dfs.add(df_new_signature)
#
#                         # Calculate cost using the *CURRENT* (and potentially updated) PCFG
#                         new_cost = pcfg.get_cost(new_sequence)
#
#                         # Push the new sequence to the priority queue
#                         heapq.heappush(pq, (new_cost, new_sequence, new_mask))
#
#         print("\nFAILED TO FIND SOLUTION\n")
#         print(f"run took {time.time() - start_time:.2f} seconds")
#         print(f"checked {steps} combinations")
#         print(f"distances from ATE (with time):\n{distances_at_time_from_target}", flush=True)
#         return "Search failed to find a solution within max steps.", time.time() - start_time
import math
import time
from typing import List, Callable

import pandas as pd

from search_methods.ATE_search import ATESearch
from utils import get_moves_and_moveBit, apply_data_preparations_seq, calculate_ate_linear_regression_lstsq, \
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

                    #TODO: check bit mask... should be better?
                    if any(f_n.split("_")[0] == func_name.split("_")[0] for f_n, c in seq  if c == col):
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
                        print(f"PROBE TRIGGERED! Error reduced from {best_ate_error:.3f} to {current_error:.3f} (ATE went to {new_ate}).")
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
