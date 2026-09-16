import math
import time
from collections import deque
from typing import List, Callable

import pandas as pd

from search_methods.ATE_search import ATESearch
from utils import df_signature_fast, apply_data_preparations_seq, get_baseline_ate, \
    calculate_ate_linear_regression_lstsq, get_moves_and_moveBit, calculate_ate_with_uncertainty, \
    analyze_ate_search_space, get_fill_combinations


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
    def search(self,
               df: pd.DataFrame,
               common_causes: List[str],
               target_ate: float,
               epsilon: float,
               transformations_dict: dict[str, Callable],
               time_out_sec: int = 14400,
               ops_by_type=None,
               fill_by_type=None,
               seq_ates_writer=None):
        col_types = df.attrs.get('col_types', None)
        df_ = df.copy()


        # print(calculate_ate_with_uncertainty(df.copy(), 'treatment', 'outcome', common_causes))
        if df.isnull().values.any():  # Need to fill
            fill_methods = get_fill_combinations(df, col_types, fill_by_type)
            print(f"{len(fill_methods)} options to fill missing")
            seen_dfs = set()
            Q = deque()
            smallest_distance_from_target = None
            best_init = None
            prune_count = 0
            num_prog_seen = 0
            for fill_sequence in fill_methods:
                num_prog_seen += 1
                filled_df = apply_data_preparations_seq(df_, fill_sequence, transformations_dict)
                filled_sig = df_signature_fast(filled_df, common_causes)
                if filled_sig in seen_dfs:  #doing OE on the fill sequences
                    prune_count += 1
                    continue
                seen_dfs.add(filled_sig)
                Q.append((fill_sequence,0))
                new_ate = calculate_ate_linear_regression_lstsq(filled_df, 'treatment', 'outcome',
                                                                common_causes)
                if smallest_distance_from_target is None or abs(new_ate - target_ate) < smallest_distance_from_target:
                    smallest_distance_from_target = abs(new_ate - target_ate)
                    best_init = fill_sequence
                if seq_ates_writer is not None:
                    seq_ates_writer.writerow((fill_sequence, new_ate))
        else:
            baseline_ate = get_baseline_ate(common_causes, df_)
            print(f"start ATE is {baseline_ate}")
            Q = deque([((), 0)])
            if seq_ates_writer is not None:
                seq_ates_writer.writerow(((), baseline_ate))
            seen_dfs = set()
            seen_dfs.add(df_signature_fast(df_, common_causes))
            prune_count = 0
            num_prog_seen = 0
            smallest_distance_from_target = abs(baseline_ate - target_ate)
            best_init = ()
        solution_seq = None
        # run_times = []
        # run_times_pop = []
        Q_poped_num = 0
        fast_moves = get_moves_and_moveBit(common_causes, transformations_dict.keys(), col_types, ops_by_type)
        found_solution = False

        print("Init complete")

        distances_at_time_from_target = [(smallest_distance_from_target, 0)]

        start_time = time.time()

        if smallest_distance_from_target < epsilon:
            solution_seq = best_init
            found_solution = True
            print("FOUND SOLUTION WITH NO NEED OF DATA PREP")

        while len(Q) > 0 and not found_solution:
            if time.time() - start_time > time_out_sec:
                print("\n\n*** TIMED OUT!! ***\n")
                break
            # start_pop_Q_time = time.time()
            Q_poped_num += 1
            seq_arr, mask = Q.popleft()
            curr_df = apply_data_preparations_seq(df_, seq_arr, transformations_dict)

            for func_name, col, move_bit in fast_moves:
                if time.time() - start_time > time_out_sec:
                    print("\n\n*** TIMED OUT!! ***\n")
                    break
                if found_solution:
                    break

                # col_type = col_types.get(col) if col_types else None
                if func_name == "isolationForest" and any(f_n == "isolationForest" for f_n, c in seq_arr):
                    continue
                if func_name == "drop_duplicates" and any(f_n == "drop_duplicates" for f_n, c in seq_arr):
                    continue
                if mask & move_bit:
                    continue
                # time_col_func_start = time.time()
                num_prog_seen += 1
                new_df = transformations_dict[func_name](curr_df.copy(), col)
                df_new_signature = df_signature_fast(new_df, common_causes)

                if df_new_signature in seen_dfs:
                    prune_count += 1

                # if df hasnt been explored:
                else:
                    new_ate = calculate_ate_linear_regression_lstsq(new_df, 'treatment', 'outcome',
                                                                    common_causes)
                    new_path = seq_arr + ((func_name, col),)
                    if seq_ates_writer is not None:
                        seq_ates_writer.writerow((new_path, new_ate))

                    new_distance = abs(new_ate - target_ate)
                    if new_distance < smallest_distance_from_target:
                        smallest_distance_from_target = new_distance
                        distances_at_time_from_target.append((new_distance, time.time() - start_time))
                    if abs(new_ate - target_ate) < epsilon:
                        solution_seq = new_path
                        print(
                            f"""***\n\nFINISHED\nATE now is: {new_ate}\nsequence is: {solution_seq}\n***""",
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

                # time_col_func_end = time.time()
                # run_times.append(time_col_func_end - time_col_func_start) #appears to be unused

            end_pop_Q_time = time.time()
            # run_times_pop.append(end_pop_Q_time - start_pop_Q_time) #this appears to be unused
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
        return {"solution_seq": solution_seq}
