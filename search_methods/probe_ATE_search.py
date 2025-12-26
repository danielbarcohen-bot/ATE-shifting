import heapq
import time
from typing import List

import numpy as np
import pandas as pd

from search_methods.ATE_search import ATESearch
from search_methods.pruning_ATE_search import canonical
from utils import get_transformations, apply_data_preparations_seq, calculate_ate_linear_regression_lstsq, \
    df_signature_fast


# --- PCFG Class to Manage Weights ---

class PCFG:
    def __init__(self, operations, columns):
        self.rules = {}  # Stores {rule_name: probability}
        self.non_terminals = {}  # Stores {non_terminal: [rule_names]}
        self._initialize_weights(operations, columns)

    def _initialize_weights(self, operations, columns):
        # 1. Operation Selection Rules (Non-terminal O)
        op_rules = []
        for op in operations:
            for col in columns:
                rule_name = f"{op}_{col}"
                self.rules[rule_name] = 1.0 / (len(operations) * len(columns))  # Uniform initial weight
                op_rules.append(rule_name)
        self.non_terminals['O'] = op_rules

        # 2. Sequence Continuation/Termination Rules (Non-terminal S)
        # S -> O S (Continue)
        self.rules['S_continue'] = 0.8
        # S -> E (End)
        self.rules['S_end'] = 0.2
        self.non_terminals['S'] = ['S_continue', 'S_end']

    def get_prob(self, rule_name):
        """Get the probability of a specific rule."""
        return self.rules.get(rule_name, 0.0)

    def get_cost(self, sequence):
        """Calculates -log(Probability) for the entire sequence."""
        if not sequence:
            return -np.log(self.rules['S_end'])  # Cost of just terminating

        # Start with the probability of the first operation
        log_prob = 0.0

        # Add probability of each operation choice and sequence continuation
        for op, col in sequence:
            # P(O) for the specific operation choice
            rule_name = f"{op}_{col}"
            log_prob += np.log(self.rules.get(rule_name, 1e-10))
            # P(S -> O S) for sequence continuation (except the last one)
            log_prob += np.log(self.rules.get('S_continue', 1e-10))

        # Add probability of termination
        log_prob += np.log(self.rules.get('S_end', 1e-10))

        return -log_prob / (len(sequence) + 1)  # -log_prob  # Cost = -log(P)

    def update_weights(self, probe_sequence, alpha=0.2):
        """
        Updates weights using the interpolation method based on a successful probe.
        """
        # Calculate the empirical distribution D_probe from the sequence
        rule_counts = {}
        for op, col in probe_sequence:
            rule_name = f"{op}_{col}"
            rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1

        # 1. Update Operation Selection Rules ('O')
        for rule_name in self.non_terminals['O']:
            # D_probe is 1 if the rule was used, 0 otherwise (in this simplified view)
            is_used = 1.0 if rule_name in rule_counts else 0.0

            # Interpolation: W_new = (1-a)*W_current + a*W_probe
            self.rules[rule_name] = (1.0 - alpha) * self.rules[rule_name] + alpha * is_used

        # 2. Normalize the Operation Selection Rules (must sum to 1)
        # This is a critical step: ensure the weights of competing rules still sum to 1.
        total_prob = sum(self.rules[r] for r in self.non_terminals['O'])
        for r in self.non_terminals['O']:
            self.rules[r] /= total_prob

        # Optional: You could update S_continue/S_end if you want to learn sequence length,
        # but we keep them fixed here for simplicity.


class ProbeATESearch(ATESearch):
    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
               max_seq_length: int):
        """
        Implements the adaptive Best-First Search guided by the PCFG cost.
        """
        pcfg = PCFG([func_name for func_name, func in get_transformations().items()], common_causes)
        # Priority Queue stores tuples: (cost, sequence)
        # The cost is the PCFG cost, NOT the ATE error.
        pq = []
        transformations_dict = get_transformations().items()

        # 1. Start with the empty sequence (Cost calculated from PCFG)
        initial_sequence = []
        initial_cost = pcfg.get_cost(initial_sequence)
        heapq.heappush(pq, (initial_cost, initial_sequence))

        steps = 0
        best_ate_error = float('inf')

        seen_seq = set()
        seen_dfs = set()
        seen_dfs.add(df_signature_fast(df.copy(), common_causes))

        start_time = time.time()

        while pq:
            cost, sequence = heapq.heappop(pq)
            steps += 1

            # --- ATE Evaluation & Goal Check ---
            curr_df = apply_data_preparations_seq(df, sequence)
            curr_df_filled = curr_df.dropna()  # curr_df.fillna(value=curr_df.mean())
            current_ate = calculate_ate_linear_regression_lstsq(curr_df_filled, 'treatment', 'outcome', common_causes)
            current_error = abs(current_ate - target_ate)

            # print(f"Step {steps:03d} | Sequence: {sequence} | Cost: {cost:.2f} | ATE: {current_ate:.3f} | Error: {current_error:.3f}")

            if current_error < epsilon:
                print(f"\n🏆 **GOAL REACHED!** Final Sequence: {sequence} with ATE {current_ate:.7f}")
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

                    # 1. Extract all items from the current queue
                    current_frontier = []
                    while pq:
                        # We only stored (cost, sequence). We need to extract the sequence.
                        old_cost, seq = heapq.heappop(pq)
                        current_frontier.append(seq)

                    # 2. Re-calculate the cost for every sequence using the updated PCFG
                    for seq in current_frontier:
                        new_cost = pcfg.get_cost(seq)
                        # 3. Push the sequence back with the new, correct cost
                        heapq.heappush(pq, (new_cost, seq))

                    print(f"✅ Frontier refreshed. Queue size: {len(pq)}")

            # --- Expansion (Generating Children) ---
            # Generate new sequences by appending one operation
            if len(sequence) < max_seq_length:
                for func_name, func in transformations_dict:
                    for col in common_causes:
                        new_op = (func_name, col)
                        new_sequence = sequence + [new_op]

                        if not (func_name.startswith("fill_") and curr_df[col].isna().sum() > 0) and not (
                                func_name.startswith("bin_") and any(
                            curr_func_name.startswith("bin_") and curr_col == col for curr_func_name, curr_col in
                            sequence)) and not (
                                func_name.startswith("zscore_clip") and any(
                            curr_func_name.startswith("zscore_clip") and curr_col == col for curr_func_name, curr_col in
                            sequence)):
                            new_col = func(curr_df[col].copy())

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
                                heapq.heappush(pq, (new_cost, new_sequence))

        print(f"run took {time.time() - start_time:.2f} seconds")
        print(f"checked {steps} combinations")
        return "Search failed to find a solution within max steps."

# --- Run the Search ---
# if __name__ == '__main__':
#     from data_loader import TwinsDataLoader
#     from utils import get_transformations, apply_data_preparations_seq, calculate_ate_linear_regression_lstsq
#
#     df = TwinsDataLoader().load_data()
#     common_causes = df.columns.difference(["treatment", "outcome"]).tolist()#['wt', 'hydra', 'nprevistq', 'gestat10']
#     target_ate = 0.003#0.005
#     epsilon = 0.0033#0.0001
#     max_length = 10#5
#     print(f"--- Starting JIT ATE Search (Target ATE: {target_ate} +/- {epsilon}) ---")
#     pcfg = PCFG([func_name for func_name, func in get_transformations().items()], common_causes)
#     result = jit_ate_search(df, target_ate, epsilon, pcfg, max_length, common_causes)
#     print(result)
