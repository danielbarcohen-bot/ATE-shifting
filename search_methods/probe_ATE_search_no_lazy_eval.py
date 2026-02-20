import time
import heapq
from typing import List, Callable

import pandas as pd

from search_methods.probe_ATE_search import ProbeATESearch
from search_methods.pruning_ATE_search import canonical
from utils import get_moves_and_moveBit, df_signature_fast, calculate_ate_linear_regression_lstsq, \
    apply_data_preparations_seq


class ProbeATESearchNoLazyEval(ProbeATESearch):
    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
               max_seq_length: int, transformations_dict: dict[str, Callable]):
        """
        Implements the adaptive Best-First Search guided by the PCFG cost.
        """
        pcfg = None#PCFG([func_name for func_name, func in transformations_dict.items()], common_causes)
        fast_moves = get_moves_and_moveBit(common_causes, transformations_dict.keys())
        # Priority Queue stores tuples: (cost, sequence)
        # The cost is the PCFG cost, NOT the ATE error.
        pq = []
        # 1. Start with the empty sequence (Cost calculated from PCFG)
        initial_sequence = ()
        initial_cost = pcfg.get_cost(initial_sequence)
        heapq.heappush(pq, (initial_cost, initial_sequence, 0, calculate_ate_linear_regression_lstsq(df.copy(), 'treatment', 'outcome', common_causes)))

        steps = 0
        best_ate_error = float('inf')

        seen_seq = set()
        seen_dfs = set()
        seen_dfs.add(df_signature_fast(df.copy(), common_causes))

        start_time = time.time()


        while pq:
            cost, sequence, mask, current_ate = heapq.heappop(pq)
            steps += 1

            # --- ATE Evaluation & Goal Check ---
            curr_df = apply_data_preparations_seq(df, sequence, transformations_dict)
            curr_df_filled = curr_df  # .dropna()  # curr_df.fillna(value=curr_df.mean())
            # current_ate = calculate_ate_linear_regression_lstsq(curr_df_filled, 'treatment', 'outcome', common_causes)
            current_error = abs(current_ate - target_ate)



            if current_error < epsilon:
                print(f"\n **GOAL REACHED!** Final Sequence: {sequence} with ATE {current_ate:.7f}")
                print(f"run took {time.time() - start_time:.2f} seconds")
                print(f"checked {steps} combinations")
                return sequence

            # --- Probe Trigger (JIT Learning) ---
            # A probe is triggered if this sequence is significantly better than the best error seen so far.
            # Here we use a simple check: if the error is reduced by 50%
            if current_error < best_ate_error * 0.9:  # * 0.5:
                print(
                    f"🔥 PROBE TRIGGERED! Error reduced from {best_ate_error:.3f} to {current_error:.3f} (ATE went to {current_ate}).")
                pcfg.update_weights(sequence)
                best_ate_error = current_error  # Update the best error seen
                # reset the pq:
                if pq:
                    print("💡 Re-scoring search frontier...")
                    # restart the smallest \ largest ive seen


                    # 1. Extract all items from the current queue
                    current_frontier = []
                    while pq:
                        # We only stored (cost, sequence). We need to extract the sequence.
                        old_cost, seq, mask_, ate = heapq.heappop(pq)
                        current_frontier.append((seq, mask_, ate))

                    # 2. Re-calculate the cost for every sequence using the updated PCFG
                    for seq, mask_, ate in current_frontier:
                        new_cost = pcfg.get_cost(seq)
                        # 3. Push the sequence back with the new, correct cost
                        heapq.heappush(pq, (new_cost, seq, mask_, ate))

                    print(f"✅ Frontier refreshed. Queue size: {len(pq)}")

            # --- Expansion (Generating Children) ---
            # Generate new sequences by appending one operation
            if len(sequence) < max_seq_length:
                for func_name, col, move_bit in fast_moves:
                    new_mask = mask | move_bit
                    new_sequence = sequence + ((func_name, col),)

                    # Q.append((new_path, new_mask))

                    if mask & move_bit or (func_name.startswith("fill_") and curr_df[col].isna().sum() == 0):
                        continue
                    new_col = transformations_dict[func_name](curr_df[col].copy())

                    canonical_sequence = canonical(new_sequence)
                    if canonical_sequence in seen_seq or new_col.equals(
                            curr_df[col]):  # col hasn't changed - so same df!
                        continue

                    seen_seq.add(canonical_sequence)
                    new_df = curr_df.copy(deep=False)
                    new_df[col] = new_col
                    df_new_signature = df_signature_fast(new_df, common_causes)

                    if df_new_signature in seen_dfs:
                        continue

                    # if df hasnt been explored:
                    else:
                        # print(f"added func {func_name}", flush=True)
                        seen_dfs.add(df_new_signature)

                        # Calculate cost using the *CURRENT* (and potentially updated) PCFG
                        new_cost = pcfg.get_cost(new_sequence)

                        # Push the new sequence to the priority queue
                        heapq.heappush(pq, (new_cost, new_sequence, new_mask, calculate_ate_linear_regression_lstsq(new_df, 'treatment', 'outcome', common_causes)))

        print("\nFAILED TO FIND SOLUTION\n")
        print(f"run took {time.time() - start_time:.2f} seconds")
        print(f"checked {steps} combinations")
        return "Search failed to find a solution within max steps."
