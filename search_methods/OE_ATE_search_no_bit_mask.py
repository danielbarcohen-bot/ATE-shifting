import time
from collections import deque
from typing import List, Callable

import pandas as pd

from search_methods.ATE_search import ATESearch
from utils import df_signature_fast, apply_data_preparations_seq, get_base_line, \
    calculate_ate_linear_regression_lstsq


def canonical(seq):
    # seq: list of tuples [(col_idx, op_idx), ...]
    # we want per-column sequences
    col_dict = {}
    for op, col in seq:
        if col not in col_dict:
            col_dict[col] = []
        col_dict[col].append(op)
    # sort by column index
    key = tuple(sorted((col, tuple(ops)) for col, ops in col_dict.items()))
    return key


class OEATESearchNoBitMask(ATESearch):
    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
               max_seq_length: int, transformations_dict: dict[str, Callable]):
        df_ = df.copy()

        base_line_ate = get_base_line(common_causes, df_)
        print(f"start ATE is {base_line_ate}")
        Q = deque([()])
        seen_dfs = set()
        seen_seq = set()  # ONLY TRUE WHEN OPERATIONS AFFECT 1 COL AT A TIME
        prune_count = 0
        try_count = 0
        solution_seq = None
        seq_ates = []
        run_times = []
        run_times_pop = []
        Q_poped_num = 0
        transformations_dict_items = transformations_dict.items()
        found_solution = False

        # smallest_ate = np.inf
        # largest_ate = -np.inf
        # largest_ate_and_time_list = []
        # smallest_ate_and_time_list = []
        smallest_distance_from_target = abs(base_line_ate - target_ate)
        distances_at_time_from_target = [(smallest_distance_from_target, 0)]

        start_time = time.time()
        while len(Q) > 0:
            if found_solution:
                break
            start_pop_Q_time = time.time()
            Q_poped_num += 1
            seq_arr = Q.popleft()
            curr_df = apply_data_preparations_seq(df_, seq_arr, transformations_dict)
            curr_df_filled = curr_df
            new_ate = calculate_ate_linear_regression_lstsq(curr_df_filled, 'treatment', 'outcome',
                                                            common_causes)
            seq_ates.append((seq_arr, new_ate))

            new_distance = abs(new_ate - target_ate)
            if new_distance < smallest_distance_from_target:
                smallest_distance_from_target = new_distance
                distances_at_time_from_target.append((new_distance, time.time() - start_time))

            if abs(new_ate - target_ate) < epsilon:
                print(
                    f"""***\nFINISHED\nATE before: {base_line_ate}\nATE now is: {new_ate}\nsequence is: {seq_arr}\n***""",
                    flush=True)
                solution_seq = seq_arr
                break

            if len(seq_arr) < max_seq_length:
                for col in common_causes:
                    if found_solution:
                        break
                    for func_name, func in transformations_dict_items:
                        if found_solution:
                            break
                        try_count += 1
                        if any(past_op.split('_')[0] == func_name.split('_')[0] and past_col == col for
                               past_op, past_col in seq_arr):
                            continue  # the group op (bin \ norm) already done to this col
                        time_col_func_start = time.time()
                        # new_df = curr_df.copy()
                        # new_df[col] = func(new_df[col])
                        new_col = transformations_dict[func_name](curr_df[col].copy())

                        canonical_sequence = canonical(seq_arr + ((func_name, col),))
                        if canonical_sequence in seen_seq or new_col.equals(
                                curr_df[col]):  # col hasn't changed - so same df!
                            prune_count += 1
                            continue

                        seen_seq.add(canonical_sequence)
                        new_df = curr_df.copy(deep=False)
                        new_df[col] = new_col
                        df_new_signature = df_signature_fast(new_df, common_causes)

                        if df_new_signature in seen_dfs:
                            prune_count += 1

                        # if df hasnt been explored:
                        else:
                            # print(f"added func {func_name}", flush=True)
                            new_ate = calculate_ate_linear_regression_lstsq(new_df.copy(), 'treatment', 'outcome',
                                                                            common_causes)
                            if abs(new_ate - target_ate) < epsilon:
                                solution_seq = seq_arr + ((func_name, col),)
                                print(
                                    f"""***\n\nFINISHED\nATE before: {base_line_ate}\nATE now is: {new_ate}\nsequence is: {solution_seq}\n***""",
                                    flush=True)

                                found_solution = True
                                break
                            seen_dfs.add(df_new_signature)
                            new_path = seq_arr + ((func_name, col),)

                            Q.append(new_path)
                            # Q.append(seq_arr + [(func_name, col)])

                        time_col_func_end = time.time()
                        run_times.append(time_col_func_end - time_col_func_start)

            end_pop_Q_time = time.time()
            run_times_pop.append(end_pop_Q_time - start_pop_Q_time)
        end_time = time.time()
        execution_time = end_time - start_time
        if solution_seq is None:
            print("*** Didn't find solution! ***")
        print(f"Execution time: {execution_time} seconds", flush=True)
        print(f"pruned {prune_count}", flush=True)
        print(f"popped from Q {Q_poped_num} nodes", flush=True)

        print(f"checked {try_count} combinations", flush=True)

        print(f"distances from ATE (with time):\n{distances_at_time_from_target}", flush=True)

        return solution_seq
