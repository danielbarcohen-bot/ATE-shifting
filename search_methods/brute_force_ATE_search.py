import math
import time
from collections import deque
from typing import List, Callable

import numpy as np
import pandas as pd

from search_methods.ATE_search import ATESearch
from search_methods.probe_ATE_search import ProbManager
from utils import apply_data_preparations_seq, get_baseline_ate, \
    calculate_ate_linear_regression_lstsq, get_moves_and_moveBit


class BruteForceATESearch(ATESearch):

    def __init__(self, op_probs=None):
        self.op_probs = op_probs

    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
               transformations_dict: dict[str, Callable], time_out_sec: int = 14400, whole_df_ops: List[str] = None):
        df_ = df.copy()
        # Precompute function prefixes once at startup
        func_prefixes = {func_name: func_name.split("_")[0] for func_name in transformations_dict.keys()}

        base_line_ate = get_baseline_ate(common_causes, df_)
        print(f"base_line_ate: {base_line_ate}")
        Q = deque([()])
        try_count = 0
        solution_seq = None
        seq_ates = []
        reached_goal_sequences = []  # for prob mode
        prob_manager = None if self.op_probs is None else ProbManager(
            [func_name for func_name, func in transformations_dict.items()], common_causes, self.op_probs,
            whole_df_ops=whole_df_ops)
        whole_df_ops = whole_df_ops or []
        i = 0
        transformations_dict_keys = transformations_dict.keys()

        start_time = time.time()
        while len(Q) > 0:
            if time.time() - start_time > time_out_sec:
                print("\n\n*** TIMED OUT!! ***\n")
                break
            i += 1
            # seq_arr, mask = Q.popleft()
            seq_arr = Q.popleft()
            curr_df = apply_data_preparations_seq(df_, seq_arr, transformations_dict)
            new_ate = calculate_ate_linear_regression_lstsq(curr_df, 'treatment', 'outcome',
                                                            common_causes)
            seq_ates.append((seq_arr, new_ate))

            if abs(new_ate - target_ate) < epsilon:
                if self.op_probs is not None:
                    reached_goal_sequences.append(seq_arr)
                else:
                    solution_seq = seq_arr
                    print(
                        f"""***\nFINISHED\nATE before: {base_line_ate}\nATE now is: {new_ate}\nsequence is: {seq_arr}\n***""",
                        flush=True)
                    break

            # for func, col, move_bit in fast_moves:
            for func_name in transformations_dict_keys:
                if func_name in whole_df_ops:
                    if any(f_n == func_name for f_n, c in seq_arr):
                        continue
                    try_count += 1
                    new_path = seq_arr + ((func_name, "TABLE"),)
                    Q.append(new_path)
                else:
                    for col in common_causes:
                        curr_prefix = func_prefixes[func_name]
                        if any(func_prefixes[f_n] == curr_prefix for f_n, c in seq_arr if c == col):
                            continue
                        try_count += 1

                        new_path = seq_arr + ((func_name, col),)
                        Q.append(new_path)

        end_time = time.time()
        execution_time = end_time - start_time

        if self.op_probs is not None:
            if len(reached_goal_sequences) > 0:
                most_probable_sequence = reached_goal_sequences[
                    np.argmax([prob_manager.get_sequence_probability(sequence) for sequence in reached_goal_sequences])]
                transformed_df = apply_data_preparations_seq(df_, most_probable_sequence, transformations_dict)
                new_ate = calculate_ate_linear_regression_lstsq(transformed_df, 'treatment', 'outcome', common_causes)
                print(
                    f"""***\nFINISHED\nATE before: {base_line_ate}\nATE now is: {new_ate}\n***""",
                    flush=True)
                print(f"Most probable sequence: {most_probable_sequence}")
                print(
                    f"probability of this sequence is: {np.max([prob_manager.get_sequence_probability(sequence) for sequence in reached_goal_sequences])}")
            else:
                print(f"No reached goal sequence")
        print(f"Execution time: {execution_time} seconds", flush=True)
        print(f"popped {i} from Q", flush=True)
        print(f"checked {try_count} combinations", flush=True)
        # print(
        #     f"run time per neighbor: mean: {np.mean(run_times)}, percentiles={np.percentile(run_times, [25, 75, 90, 95, 99]).tolist()}",
        #     flush=True)
        # print(f"all ates: {sorted(seq_ates, key=lambda x: len(x[0]))}", flush=True)

        return solution_seq
