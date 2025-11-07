import time
from collections import deque
from typing import List

import pandas as pd
from dowhy import CausalModel

from fast_dowhy_ATE import FastDoWhyATE
from utils import calculate_ate, get_transformations, df_signature_fast, apply_data_preparations_seq


class ATESearch(object):
    pass


class PruneATESearch(ATESearch):

    # def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
    #            max_seq_length: int):
    #     # base_line_ate = get_base_line(common_causes, df)
    #     df_ = df.copy()
    #     Q = deque([([], df_)])
    #     seen_dfs = set()
    #     prune_count = 0
    #     try_count = 0
    #     solution_seq = None
    #     seq_ates = []
    #
    #     start_time = time.time()
    #     while len(Q) > 0:
    #         seq_arr, curr_df = Q.popleft()
    #         print(f"sequence poped is: {seq_arr}", flush=True)
    #         curr_df_filled = curr_df.fillna(value=curr_df.mean())
    #         model = CausalModel(data=curr_df_filled, treatment='treatment', outcome='outcome',
    #                             common_causes=common_causes)
    #         new_ate = calculate_ate(model)
    #         seq_ates.append((seq_arr, new_ate))
    #
    #         print(f"new ate is {new_ate}", flush=True)
    #         if abs(new_ate - target_ate) < epsilon:
    #             print(f"""***\nFINISHED\nATE now is:{new_ate}\nsequence is:{seq_arr}\n***""", flush=True)
    #             solution_seq = seq_arr
    #             break
    #
    #         if len(seq_arr) < max_seq_length:
    #             for col in common_causes:
    #                 for func_name, func in get_transformations().items():
    #                     if func_name not in [f_name for (f_name, c) in seq_arr if c == col]:
    #                         print(f"checking col {col} and func {func_name}", flush=True)
    #                         time_col_func_start = time.time()
    #                         try_count += 1
    #                         new_df = curr_df.copy()
    #                         new_df[col] = func(new_df[col])
    #
    #                         # check if seen before
    #                         df_new_exists = False
    #                         new_seq = seq_arr + [(func_name, col)]
    #
    #                         df_new_signature = df_signature_fast(new_df, common_causes)
    #
    #                         if df_new_signature in seen_dfs:
    #                             prune_count += 1
    #                             df_new_exists = True
    #                             print("PRUNED", flush=True)
    #
    #                         # if not df_already_exists:
    #                         if not df_new_exists:
    #                             print(f"added func {func_name}", flush=True)
    #                             seen_dfs.add(df_new_signature)
    #                             Q.append((new_seq, new_df))
    #
    #                         time_col_func_end = time.time()
    #                         print(
    #                             f"time for col {col} and func {func_name}: {time_col_func_end - time_col_func_start} seconds", flush=True)
    #
    #     end_time = time.time()
    #     execution_time = end_time - start_time
    #     print(f"Execution time: {execution_time} seconds", flush=True)
    #     print(f"checked {try_count} combinations", flush=True)
    #     print(f"all ates: {sorted(seq_ates, key=lambda x: len(x[0]))}", flush=True)
    #     return solution_seq

    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
               max_seq_length: int):
        df_ = df.copy()
        Q = deque([[]])
        seen_dfs = set()
        prune_count = 0
        try_count = 0
        solution_seq = None
        seq_ates = []
        fast_dowhy_ate = FastDoWhyATE(df_, 'treatment', 'outcome', common_causes)

        start_time = time.time()
        while len(Q) > 0:
            seq_arr = Q.popleft()
            curr_df = apply_data_preparations_seq(df_, seq_arr)
            curr_df_filled = curr_df.fillna(value=curr_df.mean())
            model = CausalModel(data=curr_df_filled, treatment='treatment', outcome='outcome',
                                common_causes=common_causes)
            new_ate = fast_dowhy_ate.calculate_ate(model)  # calculate_ate(model)
            seq_ates.append((seq_arr, new_ate))

            print(f"new ate is {new_ate}", flush=True)
            if abs(new_ate - target_ate) < epsilon:
                print(f"""***\nFINISHED\nATE now is:{new_ate}\nsequence is:{seq_arr}\n***""", flush=True)
                solution_seq = seq_arr
                break

            if len(seq_arr) < max_seq_length:
                for col in common_causes:
                    for func_name, func in get_transformations().items():
                        if func_name not in [f_name for (f_name, c) in seq_arr if c == col]:
                            print(f"checking sequence:\n{seq_arr + [(func_name, col)]}", flush=True)
                            time_col_func_start = time.time()
                            try_count += 1
                            new_df = curr_df.copy()
                            new_df[col] = func(new_df[col])

                            # check if seen before
                            df_new_exists = False

                            df_new_signature = df_signature_fast(new_df, common_causes)

                            if df_new_signature in seen_dfs:
                                prune_count += 1
                                df_new_exists = True
                                print("PRUNED", flush=True)

                            # if not df_already_exists:
                            if not df_new_exists:
                                print(f"added func {func_name}", flush=True)
                                seen_dfs.add(df_new_signature)
                                Q.append(seq_arr + [(func_name, col)])

                            time_col_func_end = time.time()
                            print(
                                f"time for col {col} and func {func_name}: {time_col_func_end - time_col_func_start} seconds",
                                flush=True)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds", flush=True)
        print(f"checked {try_count} combinations", flush=True)
        print(f"all ates: {sorted(seq_ates, key=lambda x: len(x[0]))}", flush=True)
        return solution_seq
