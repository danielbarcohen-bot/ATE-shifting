import time
from collections import deque
from typing import List

import pandas as pd
from dowhy import CausalModel

from utils import calculate_ate, get_transformations, df_signature_fast


class ATESearch(object):
    pass


class PruneATESearch(ATESearch):

    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
               max_seq_length: int):
        # base_line_ate = get_base_line(common_causes, df)
        df_ = df.copy()
        Q = deque([([], df_)])
        seen_dfs = set()
        prune_count = 0
        try_count = 0
        solution_seq = None
        ates = []

        start_time = time.time()
        while len(Q) > 0:
            seq_arr, curr_df = Q.popleft()
            print(f"sequence poped is: {seq_arr}")
            curr_df_filled = curr_df.fillna(value=curr_df.mean())
            model = CausalModel(data=curr_df_filled, treatment='treatment', outcome='outcome',
                                common_causes=common_causes)
            new_ate = calculate_ate(model)
            ates.append(new_ate)

            print(f"new ate is {new_ate}")
            if abs(new_ate - target_ate) < epsilon:
                print(f"""***\nFINISHED\nATE now is:{new_ate}\nsequence is:{seq_arr}\n***""")
                solution_seq = seq_arr
                break

            if len(seq_arr) < max_seq_length:
                for col in common_causes:
                    for func_name, func in get_transformations().items():
                        if func_name not in [f_name for (f_name, c) in seq_arr if c == col]:
                            print(f"checking col {col} and func {func_name}")
                            time_col_func_start = time.time()
                            try_count += 1
                            new_df = curr_df.copy()
                            new_df[col] = func(new_df[col])

                            # check if seen before
                            df_new_exists = False
                            new_seq = seq_arr + [(func_name, col)]

                            time_signature_start = time.time()
                            df_new_signature = df_signature_fast(new_df, common_causes)
                            time_signature_end = time.time()
                            print(f"time for df_signature: {time_signature_end - time_signature_start} seconds")

                            time_seen_df_start = time.time()
                            if df_new_signature in seen_dfs:
                                prune_count += 1
                                df_new_exists = True
                                print("PRUNED")

                            time_seen_df_end = time.time()
                            print(f"time for checking seen_df: {time_seen_df_end - time_seen_df_start} seconds")

                            # if not df_already_exists:
                            if not df_new_exists:
                                print(f"added func {func_name}")
                                seen_dfs.add(df_new_signature)
                                Q.append((new_seq, new_df))

                            time_col_func_end = time.time()
                            print(
                                f"time for col {col} and func {func_name}: {time_col_func_end - time_col_func_start} seconds")

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")
        print(f"checked {try_count} combinations")
        print(f"all ates: {sorted(ates)}")
        return solution_seq
