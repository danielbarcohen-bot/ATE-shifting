# from sklearn.linear_model import LinearRegression
# import heapq
# import time
# from typing import List, Callable
#
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
#
# from search_methods.ATE_search import ATESearch
# from search_methods.probe_ATE_search import PCFG
# from search_methods.pruning_ATE_search import canonical
# from utils import apply_data_preparations_seq, calculate_ate_linear_regression_lstsq, \
#     df_signature_fast, get_moves_and_moveBit
#
#
# class CausalHeuristicPCFG(PCFG):
#     def __init__(self, df, treatment_col, outcome_col, operations, temperature=1.0):
#         self.df = df.copy().dropna()
#         self.treatment_col = treatment_col
#         self.outcome_col = outcome_col
#         self.temperature = temperature
#
#         # Columns used for transformations (excluding T and Y)
#         self.target_columns = [c for c in df.columns if c not in [treatment_col, outcome_col]]
#
#         super().__init__(operations, self.target_columns)
#
#     def _initialize_weights(self, operations, columns):
#         """Overrides parent to use Sensitivity Scores for initial weights."""
#         # 1. Calculate Sensitivity Scores (The Heuristic)
#         scores = self._calculate_sensitivity_scores(columns)
#
#         # 2. Softmax-like weight assignment for each column
#         # Prob(column_i) = exp(score_i / T) / sum(exp(scores / T))
#         exp_scores = np.exp(scores / self.temperature)
#         col_probs = exp_scores / np.sum(exp_scores)
#         col_prob_map = dict(zip(columns, col_probs))
#
#         # 3. Distribute probabilities across op_col combinations
#         # Prob(op_col) = Prob(column) / num_operations
#         op_rules = []
#         num_ops = len(operations)
#
#         for col in columns:
#             p_col = col_prob_map[col]
#             for op in operations:
#                 rule_name = f"{op}_{col}"
#                 # The sum of all op_col probabilities will equal 1.0
#                 self.rules[rule_name] = p_col / num_ops
#                 op_rules.append(rule_name)
#
#         self.non_terminals['O'] = op_rules
#
#         # 4. Standard Sequence Rules
#         self.rules['S_continue'] = 0.8
#         self.rules['S_end'] = 0.2
#         self.non_terminals['S'] = ['S_continue', 'S_end']
#
#     def _calculate_sensitivity_scores(self, columns):
#         """The core heuristic logic: |Beta_std| * |Corr(T, X)|"""
#         X = self.df[columns]
#         T = self.df[self.treatment_col]
#         y = self.df[self.outcome_col]
#
#         # Standardize for coefficient comparison
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
#
#         # Regression: Y ~ T + X_scaled
#         X_design = np.column_stack((T, X_scaled))
#         model = LinearRegression().fit(X_design, y)
#
#         # Extract Standardized Betas (ignore intercept and T)
#         betas_std = np.abs(model.coef_[1:])
#
#         # Correlations with Treatment
#         correlations = np.array([abs(np.corrcoef(self.df[c], T)[0, 1]) for c in columns])
#
#         # Sensitivity Score
#         return (betas_std * correlations) + 1e-6
#
#
# class ProbeATESearchLinearRegHeuristic(ATESearch):
#     def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
#                max_seq_length: int, transformations_dict: dict[str, Callable]):
#         """
#         Implements the adaptive Best-First Search guided by the PCFG cost.
#         """
#         pcfg = CausalHeuristicPCFG(df, 'treatment', 'outcome',[func_name for func_name, func in transformations_dict.items()])
#         fast_moves = get_moves_and_moveBit(common_causes, transformations_dict.keys())
#
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
#             # print(f"Step {steps:03d} | Sequence: {sequence} | Cost: {cost:.2f} | ATE: {current_ate:.3f} | Error: {current_error:.3f}")
#
#             if current_error < epsilon:
#                 print(f"\n🏆 **GOAL REACHED!** Final Sequence: {sequence} with ATE {current_ate:.7f}")
#                 print(f"run took {time.time() - start_time:.2f} seconds")
#                 print(f"checked {steps} combinations")
#                 return sequence
#
#             # --- Probe Trigger (JIT Learning) ---
#             # A probe is triggered if this sequence is significantly better than the best error seen so far.
#             # Here we use a simple check: if the error is reduced by 50%
#             if current_error < best_ate_error * 0.9:  # * 0.5:
#                 print(
#                     f"🔥 PROBE TRIGGERED! Error reduced from {best_ate_error:.3f} to {current_error:.3f} (ATE went to {current_ate}).")
#                 pcfg.update_weights(sequence)
#                 best_ate_error = current_error  # Update the best error seen
#                 # reset the pq:
#                 if pq:
#                     print("💡 Re-scoring search frontier...")
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
#         print(f"run took {time.time() - start_time:.2f} seconds")
#         print(f"checked {steps} combinations")
#         return "Search failed to find a solution within max steps."
