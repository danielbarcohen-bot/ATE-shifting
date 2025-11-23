import time
from collections import deque
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from itertools import islice
from typing import List

import numpy as np
import pandas as pd
from dowhy import CausalModel

from fast_dowhy_ATE import FastDoWhyATE
from utils import get_transformations, df_signature_fast, apply_data_preparations_seq, get_base_line, \
    df_signature_fast_rounds


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

        base_line_ate = get_base_line(common_causes, df_)
        Q = deque([[]])
        seen_dfs = set()
        prune_count = 0
        try_count = 0
        solution_seq = None
        seq_ates = []
        run_times = []
        run_times_pop = []
        Q_poped_num = 0

        fast_dowhy_ate = FastDoWhyATE(df_, 'treatment', 'outcome', common_causes)

        start_time = time.time()
        while len(Q) > 0:
            start_pop_Q_time = time.time()
            Q_poped_num += 1
            seq_arr = Q.popleft()
            curr_df = apply_data_preparations_seq(df_, seq_arr)
            curr_df_filled = curr_df.fillna(value=curr_df.mean())
            model = CausalModel(data=curr_df_filled, treatment='treatment', outcome='outcome',
                                common_causes=common_causes)
            new_ate = fast_dowhy_ate.calculate_ate(model)  # calculate_ate(model)
            seq_ates.append((seq_arr, new_ate))

            if abs(new_ate - target_ate) < epsilon:
                print(
                    f"""***\nFINISHED\nATE before: {base_line_ate}\nATE now is: {new_ate}\nsequence is: {seq_arr}\n***""",
                    flush=True)
                solution_seq = seq_arr
                break

            if len(seq_arr) < max_seq_length:
                for col in common_causes:
                    for func_name, func in get_transformations().items():
                        if func_name not in [f_name for (f_name, c) in seq_arr if c == col] or (func_name.startswith("fill_") and curr_df[col].isna().sum() > 0): # remove???
                            time_col_func_start = time.time()
                            try_count += 1
                            new_df = curr_df.copy()
                            new_df[col] = func(new_df[col])

                            if new_df[col].equals(curr_df[col]): #col has'nt change - so same df!
                                prune_count += 1
                                continue

                            df_new_signature = df_signature_fast(new_df, common_causes)

                            if df_new_signature in seen_dfs:
                                prune_count += 1

                            # if df hasnt been explored:
                            else:
                                # print(f"added func {func_name}", flush=True)
                                seen_dfs.add(df_new_signature)
                                Q.append(seq_arr + [(func_name, col)])

                            time_col_func_end = time.time()
                            run_times.append(time_col_func_end - time_col_func_start)
                        else:
                            prune_count += 1

            end_pop_Q_time = time.time()
            run_times_pop.append(end_pop_Q_time - start_pop_Q_time)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds", flush=True)
        print(f"pruned {prune_count}", flush=True)
        print(f"popped from Q {Q_poped_num} nodes", flush=True)
        print(
            f"run time per popped from Q: mean: {np.mean(run_times_pop)}, percentiles={np.percentile(run_times_pop, [25, 75, 90, 95, 99]).tolist()}",
            flush=True)
        print(f"checked {try_count} combinations", flush=True)
        print(
            f"run time per neighbor: mean: {np.mean(run_times)}, percentiles={np.percentile(run_times, [25, 75, 90, 95, 99]).tolist()}",
            flush=True)
        print(f"all ates: {sorted(seq_ates, key=lambda x: len(x[0]))}", flush=True)


        return solution_seq

    def parallel_search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
                        max_seq_length: int):
        df_ = df.copy()

        base_line_ate = get_base_line(common_causes, df_)
        Q = deque([[]])
        BATCH_SIZE = 30
        seen_dfs = set()
        Q_poped_num = 0
        prune_cnt = 0
        solution_seq = None
        seq_ates = []
        run_times_pop = []
        fast_dowhy_ate = FastDoWhyATE(df_, 'treatment', 'outcome', common_causes)

        start_time = time.time()
        with ProcessPoolExecutor(max_workers=8) as pool:
            while len(Q) > 0:
                start_pop_Q_time = time.time()
                Q_poped_num += 1
                seq_arr = Q.popleft()
                curr_df = apply_data_preparations_seq(df_, seq_arr)
                curr_df_filled = curr_df.fillna(value=curr_df.mean())
                model = CausalModel(data=curr_df_filled, treatment='treatment', outcome='outcome',
                                    common_causes=common_causes)
                new_ate = fast_dowhy_ate.calculate_ate(model)
                seq_ates.append((seq_arr, new_ate))

                # print(f"new ate is {new_ate}", flush=True)
                if abs(new_ate - target_ate) < epsilon:
                    print(
                        f"""***\nFINISHED\nATE before: {base_line_ate}\nATE now is: {new_ate}\nsequence is: {seq_arr}\n***""",
                        flush=True)
                    solution_seq = seq_arr
                    break

                if len(seq_arr) < max_seq_length:
                    tasks = []
                    for col in common_causes:
                        for func_name, func in get_transformations().items():
                            tasks.append((seq_arr, curr_df, col, func_name, func, common_causes, seen_dfs))


                    futures = [
                        pool.submit(self._process_col_func_batch, batch)
                        for batch in self.batched(tasks, BATCH_SIZE)
                    ]
                    for f in futures:#as_completed(futures):
                        for res in f.result():
                            if res is None:
                                prune_cnt += 1
                                continue
                            new_seq_arr, df_sig = res
                            if df_sig not in seen_dfs:
                                seen_dfs.add(df_sig)
                                Q.append(new_seq_arr)
                            else:
                                prune_cnt += 1

                end_pop_Q_time = time.time()
                run_times_pop.append(end_pop_Q_time - start_pop_Q_time)
        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Execution time: {execution_time} seconds", flush=True)
        print(f"pruned {prune_cnt}", flush=True)
        print(f"popped from Q {Q_poped_num} nodes", flush=True)
        print(
            f"run time per popped from Q: mean: {np.mean(run_times_pop)}, percentiles={np.percentile(run_times_pop, [25, 75, 90, 95, 99]).tolist()}",
            flush=True)
        print(f"all ates: {sorted(seq_ates, key=lambda x: len(x[0]))}", flush=True)

        return solution_seq

    def _process_col_func(self, args):
        seq_arr, df_, col, func_name, func, common_causes, seen_dfs = args
        # Skip already applied transformations
        if func_name in [f_name for (f_name, c) in seq_arr if c == col]:
            return None

        # Apply the transformation
        new_df = df_.copy()
        new_df[col] = func(new_df[col])
        #df_sig = df_signature_fast(new_df, common_causes)
        df_sig = df_signature_fast_rounds(new_df, common_causes)
        # if df_sig in seen_dfs:
        #     return None

        return (seq_arr + [(func_name, col)], df_sig)

    def batched(self,iterable, n):
        it = iter(iterable)
        while True:
            batch = list(islice(it, n))
            if not batch:
                break
            yield batch

    def _process_col_func_batch(self, batch):
        results = []
        for t in batch:
            results.append(self._process_col_func(t))
        return results