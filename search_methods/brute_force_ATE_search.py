import time
from collections import deque
from typing import List

import numpy as np
import pandas as pd

from search_methods.ATE_search import ATESearch
from utils import apply_data_preparations_seq, get_transformations, get_base_line, \
    calculate_ate_linear_regression_lstsq, get_moves_and_moveBit


class BruteForceATESearch(ATESearch):

    # def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
    #            max_seq_length: int):
    #     df_ = df.copy()
    #
    #     base_line_ate = get_base_line(common_causes, df_)
    #     print(f"base_line_ate: {base_line_ate}")
    #     transformations_dict = get_transformations().items()
    #     Q = deque([[]])
    #     try_count = 0
    #     solution_seq = None
    #     seq_ates = []
    #     run_times = []
    #     # fast_dowhy_ate = FastDoWhyATE(df_, 'treatment', 'outcome', common_causes)
    #     i = 0
    #
    #     start_time = time.time()
    #     while len(Q) > 0:
    #         i += 1
    #         seq_arr = Q.popleft()
    #         curr_df = apply_data_preparations_seq(df_, seq_arr)
    #         # print(f"sequence poped is: {seq_arr}", flush=True)
    #         curr_df = curr_df.dropna()  # curr_df.fillna(value=curr_df.mean(), inplace=True)
    #         # model = CausalModel(data=curr_df, treatment='treatment', outcome='outcome', common_causes=common_causes)
    #         new_ate = calculate_ate_linear_regression_lstsq(curr_df, 'treatment', 'outcome',
    #                                                         common_causes)  # fast_dowhy_ate.calculate_ate(model)  # calculate_ate(model)
    #         seq_ates.append((seq_arr, new_ate))
    #
    #         # if len(seq_arr) > 3 and i % 50 == 0:
    #         #     print(find_interesting(seq_ates,4,6))
    #         #     print("-" * 120)
    #
    #         if abs(new_ate - target_ate) < epsilon:
    #             solution_seq = seq_arr
    #             print(
    #                 f"""***\nFINISHED\nATE before: {base_line_ate}\nATE now is: {new_ate}\nsequence is: {seq_arr}\n***""",
    #                 flush=True)
    #             break
    #
    #         if len(seq_arr) < max_seq_length:
    #             for col in common_causes:
    #                 for func_name, func in transformations_dict:
    #                     if not (
    #                             func_name.startswith("bin_") and any(
    #                         curr_func_name.startswith("bin_") and curr_col == col for curr_func_name, curr_col in
    #                         seq_arr)) and not (
    #                             func_name.startswith("zscore_clip") and any(
    #                         curr_func_name.startswith("zscore_clip") and curr_col == col for curr_func_name, curr_col in
    #                         seq_arr)):
    #                         # if func_name not in [f_name for (f_name, c) in seq_arr if c == col]:
    #                         t = time.time()
    #                         try_count += 1
    #                         # new_df = curr_df.copy()
    #                         # new_df[col] = func(new_df[col])
    #                         Q.append(seq_arr + [(func_name, col)])
    #                         run_times.append(time.time() - t)
    #
    #     end_time = time.time()
    #     execution_time = end_time - start_time
    #     print(f"Execution time: {execution_time} seconds", flush=True)
    #     print(f"checked {try_count} combinations", flush=True)
    #     print(
    #         f"run time per neighbor: mean: {np.mean(run_times)}, percentiles={np.percentile(run_times, [25, 75, 90, 95, 99]).tolist()}",
    #         flush=True)
    #     # print(f"all ates: {sorted(seq_ates, key=lambda x: len(x[0]))}", flush=True)
    #
    #     return solution_seq

    # def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
    #            max_seq_length: int):
    #     df_ = df.copy()
    #
    #     base_line_ate = get_base_line(common_causes, df_)
    #     print(f"base_line_ate: {base_line_ate}")
    #     transformations_dict = get_transformations().items()
    #     try_count = 0
    #     solution_seq = None
    #     seq_ates = []
    #     run_times = []
    #     # fast_dowhy_ate = FastDoWhyATE(df_, 'treatment', 'outcome', common_causes)
    #     i = 0
    #     context = SearchContext(get_transformations().keys(), common_causes)
    #     Q = deque([SequenceState(context)])
    #     start_time = time.time()
    #     while len(Q) > 0:
    #         i += 1
    #         seq_arr = Q.popleft()
    #         curr_df = df_.copy()#apply_data_preparations_seq(df_, seq_arr)
    #         # print(f"sequence poped is: {seq_arr}", flush=True)
    #         curr_df = curr_df.dropna()  # curr_df.fillna(value=curr_df.mean(), inplace=True)
    #         # model = CausalModel(data=curr_df, treatment='treatment', outcome='outcome', common_causes=common_causes)
    #         new_ate = calculate_ate_linear_regression_lstsq(curr_df, 'treatment', 'outcome',
    #                                                         common_causes)  # fast_dowhy_ate.calculate_ate(model)  # calculate_ate(model)
    #         seq_ates.append((seq_arr, new_ate))
    #
    #         # if len(seq_arr) > 3 and i % 50 == 0:
    #         #     print(find_interesting(seq_ates,4,6))
    #         #     print("-" * 120)
    #
    #         if abs(new_ate - target_ate) < epsilon:
    #             solution_seq = seq_arr
    #             print(
    #                 f"""***\nFINISHED\nATE before: {base_line_ate}\nATE now is: {new_ate}\nsequence is: {seq_arr}\n***""",
    #                 flush=True)
    #             break
    #
    #         if len(seq_arr.path) < max_seq_length:
    #             t = time.time()
    #             for next_state in seq_arr.expand():
    #                         # if func_name not in [f_name for (f_name, c) in seq_arr if c == col]:
    #
    #                         try_count += 1
    #                         # new_df = curr_df.copy()
    #                         # new_df[col] = func(new_df[col])
    #                         Q.append(next_state)
    #             run_times.append(time.time() - t)
    #             print(
    #                 f"expansion all possibilities time: mean: {np.mean(run_times)}, percentiles={np.percentile(run_times, [25, 75, 95, 99]).tolist()}")
    #     end_time = time.time()
    #     execution_time = end_time - start_time
    #     print(f"Execution time: {execution_time} seconds", flush=True)
    #     print(f"checked {try_count} combinations", flush=True)
    #     print(
    #         f"run time per neighbor: mean: {np.mean(run_times)}, percentiles={np.percentile(run_times, [25, 75, 90, 95, 99]).tolist()}",
    #         flush=True)
    #     # print(f"all ates: {sorted(seq_ates, key=lambda x: len(x[0]))}", flush=True)
    #
    #     return solution_seq

    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
               max_seq_length: int):
        df_ = df.copy()

        base_line_ate = get_base_line(common_causes, df_)
        print(f"base_line_ate: {base_line_ate}")
        Q = deque([((), 0)])
        try_count = 0
        solution_seq = None
        seq_ates = []
        run_times = []
        # fast_dowhy_ate = FastDoWhyATE(df_, 'treatment', 'outcome', common_causes)
        i = 0

        fast_moves = get_moves_and_moveBit(common_causes, get_transformations().keys())

        start_time = time.time()
        while len(Q) > 0:
            i += 1
            seq_arr, mask = Q.popleft()
            curr_df = apply_data_preparations_seq(df_, seq_arr)
            # print(f"sequence poped is: {seq_arr}", flush=True)
            curr_df = curr_df.dropna()  # curr_df.fillna(value=curr_df.mean(), inplace=True)
            # model = CausalModel(data=curr_df, treatment='treatment', outcome='outcome', common_causes=common_causes)
            new_ate = calculate_ate_linear_regression_lstsq(curr_df, 'treatment', 'outcome',
                                                            common_causes)  # fast_dowhy_ate.calculate_ate(model)  # calculate_ate(model)
            seq_ates.append((seq_arr, new_ate))

            # if len(seq_arr) > 3 and i % 50 == 0:
            #     print(find_interesting(seq_ates,4,6))
            #     print("-" * 120)

            if abs(new_ate - target_ate) < epsilon:
                solution_seq = seq_arr
                print(
                    f"""***\nFINISHED\nATE before: {base_line_ate}\nATE now is: {new_ate}\nsequence is: {seq_arr}\n***""",
                    flush=True)
                break

            if len(seq_arr) < max_seq_length:
                t = time.time()
                for func, col, move_bit in fast_moves:
                    try_count += 1
                    # 1. O(1) Lookup: This is roughly 100x faster than 'any()'
                    if mask & move_bit:
                        continue

                    # 2. Create new state
                    # frozenset | {id} is highly optimized in Python
                    new_mask = mask | move_bit
                    new_path = seq_arr + ((func, col),)

                    Q.append((new_path, new_mask))

                run_times.append(time.time() - t)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds", flush=True)
        print(f"checked {try_count} combinations", flush=True)
        print(
            f"run time per neighbor: mean: {np.mean(run_times)}, percentiles={np.percentile(run_times, [25, 75, 90, 95, 99]).tolist()}",
            flush=True)
        # print(f"all ates: {sorted(seq_ates, key=lambda x: len(x[0]))}", flush=True)

        return solution_seq
