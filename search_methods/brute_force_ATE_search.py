import time
from collections import deque
from typing import List

import pandas as pd
from dowhy import CausalModel

from fast_dowhy_ATE import FastDoWhyATE
from utils import apply_data_preparations_seq, calculate_ate, get_transformations


class ATESearch(object):
    pass


class BruteForceATESearch(ATESearch):

    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
               max_seq_length: int):
        # base_line_ate = get_base_line(common_causes, df)
        df_ = df.copy()
        Q = deque([[]])
        try_count = 0
        solution_seq = None
        seq_ates = []
        fast_dowhy_ate = FastDoWhyATE(df_, 'treatment', 'outcome', common_causes)
        i = 0

        start_time = time.time()
        while len(Q) > 0:
            i += 1
            seq_arr = Q.popleft()
            curr_df = apply_data_preparations_seq(df_, seq_arr)
            print(f"sequence poped is: {seq_arr}", flush=True)
            curr_df.fillna(value=curr_df.mean(), inplace=True)
            model = CausalModel(data=curr_df, treatment='treatment', outcome='outcome', common_causes=common_causes)
            new_ate = fast_dowhy_ate.calculate_ate(model)  # calculate_ate(model)
            seq_ates.append((seq_arr, new_ate))

            if i % 100 == 0:
                print(f"current checked ates: \n{sorted(seq_ates, key=lambda x: len(x[0]))}", flush=True)

            if abs(new_ate - target_ate) < epsilon:
                solution_seq = seq_arr
                print(f"""***\nFINISHED\nATE now is:{new_ate}\nsequence is:{seq_arr}\n***""", flush=True)
                break

            if len(seq_arr) < max_seq_length:
                for col in common_causes:
                    for func_name, func in get_transformations().items():
                        if func_name not in [f_name for (f_name, c) in seq_arr if c == col]:
                            try_count += 1
                            new_df = curr_df.copy()
                            new_df[col] = func(new_df[col])
                            Q.append(seq_arr + [(func_name, col)])

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds", flush=True)
        print(f"checked {try_count} combinations", flush=True)
        print(f"all ates: {sorted(seq_ates, key=lambda x: len(x[0]))}", flush=True)
        return solution_seq
