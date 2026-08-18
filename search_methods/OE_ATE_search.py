import time
from collections import deque
from typing import List, Callable

import pandas as pd

from search_methods.ATE_search import ATESearch
from utils import df_signature_fast, apply_data_preparations_seq, get_baseline_ate, \
    calculate_ate_linear_regression_lstsq, get_moves_and_moveBit, calculate_ate_with_uncertainty, \
    analyze_ate_search_space


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


class OEATESearch(ATESearch):
    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
               max_seq_length: int, transformations_dict: dict[str, Callable], time_out_sec: int = 14400):
        df_ = df.copy()

        base_line_ate = get_baseline_ate(common_causes, df_)
        print(f"start ATE is {base_line_ate}")
        print(calculate_ate_with_uncertainty(df.copy(), 'treatment', 'outcome', common_causes))
        Q = deque([((), 0)])
        seen_dfs = set()
        prune_count = 0
        num_prog_seen = 0
        solution_seq = None
        seq_ates = [((), base_line_ate)]
        run_times = []
        run_times_pop = []
        Q_poped_num = 0
        fast_moves = get_moves_and_moveBit(common_causes, transformations_dict.keys())
        found_solution = False

        smallest_distance_from_target = abs(base_line_ate - target_ate)
        distances_at_time_from_target = [(smallest_distance_from_target, 0)]

        start_time = time.time()

        if abs(base_line_ate - target_ate) < epsilon:
            solution_seq = (())
            found_solution = True
            print("FOUND SOLUTION WITH NO NEED OF DATA PREP")

        while len(Q) > 0 and not found_solution:
            if time.time() - start_time > time_out_sec:
                print("\n\n*** TIMED OUT!! ***\n")
                break
            start_pop_Q_time = time.time()
            Q_poped_num += 1
            seq_arr, mask = Q.popleft()
            curr_df = apply_data_preparations_seq(df_, seq_arr, transformations_dict)

            for func_name, col, move_bit in fast_moves:
                if time.time() - start_time > time_out_sec:
                    print("\n\n*** TIMED OUT!! ***\n")
                    break
                if found_solution:
                    break

                if func_name == "isolationForest" and any(f_n == "isolationForest" for f_n, c in seq_arr):
                    continue
                if func_name == "drop_duplicates" and any(f_n == "drop_duplicates" for f_n, c in seq_arr):
                    continue
                if mask & move_bit:
                    continue
                time_col_func_start = time.time()
                num_prog_seen += 1
                new_df = transformations_dict[func_name](curr_df.copy(), col)
                df_new_signature = df_signature_fast(new_df, common_causes)

                if df_new_signature in seen_dfs:
                    prune_count += 1

                # if df hasnt been explored:
                else:
                    new_ate = calculate_ate_linear_regression_lstsq(new_df.copy(), 'treatment', 'outcome',
                                                                    common_causes)
                    new_path = seq_arr + ((func_name, col),)
                    seq_ates.append((new_path, new_ate))

                    new_distance = abs(new_ate - target_ate)
                    if new_distance < smallest_distance_from_target:
                        smallest_distance_from_target = new_distance
                        distances_at_time_from_target.append((new_distance, time.time() - start_time))
                    if abs(new_ate - target_ate) < epsilon:
                        solution_seq = new_path
                        print(
                            f"""***\n\nFINISHED\nATE before: {base_line_ate}\nATE now is: {new_ate}\nsequence is: {solution_seq}\n***""",
                            flush=True)
                        try:
                            print(
                                f"uncertainty:\n{calculate_ate_with_uncertainty(new_df.copy(), 'treatment', 'outcome', common_causes)}")
                        except Exception as e:
                            print(f"Failed to calculate uncertainty:\n{e}")
                        found_solution = True
                        break
                    seen_dfs.add(df_new_signature)
                    new_mask = mask | move_bit

                    Q.append((new_path, new_mask))

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
        print(f"checked {num_prog_seen} combinations", flush=True)
        print(f"distances from ATE (with time):\n{distances_at_time_from_target}", flush=True)
        # return solution_seq
        return {"solution_seq": solution_seq,
                # "ates_distrb": analyze_ate_search_space(seq_ates)
                "seq_ates": seq_ates}
