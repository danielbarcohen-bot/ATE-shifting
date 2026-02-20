import time
from collections import deque
from typing import List, Callable

import numpy as np
import pandas as pd

from search_methods.ATE_search import ATESearch
from utils import df_signature_fast, apply_data_preparations_seq, get_base_line, \
    calculate_ate_linear_regression_lstsq, get_moves_and_moveBit, find_interesting


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


class PruneATESearch(ATESearch):
    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
               max_seq_length: int, transformations_dict: dict[str, Callable]):
        df_ = df.copy()

        base_line_ate = get_base_line(common_causes, df_)
        print(f"start ATE is {base_line_ate}")
        Q = deque([((), 0)])
        seen_dfs = set()
        seen_seq = set()  # ONLY TRUE WHEN OPERATIONS AFFECT 1 COL AT A TIME
        prune_count = 0
        try_count = 0
        solution_seq = None
        seq_ates = []
        run_times = []
        run_times_pop = []
        Q_poped_num = 0
        fast_moves = get_moves_and_moveBit(common_causes, transformations_dict.keys())

        # smallest_ate = np.inf
        # largest_ate = -np.inf
        # largest_ate_and_time_list = []
        # smallest_ate_and_time_list = []
        smallest_distance_from_target = abs(base_line_ate - target_ate)
        distances_at_time_from_target = [(smallest_distance_from_target, 0)]

        start_time = time.time()
        while len(Q) > 0:
            start_pop_Q_time = time.time()
            Q_poped_num += 1
            seq_arr, mask = Q.popleft()
            curr_df = apply_data_preparations_seq(df_, seq_arr, transformations_dict)
            curr_df_filled = curr_df#.dropna()
            new_ate = calculate_ate_linear_regression_lstsq(curr_df_filled, 'treatment', 'outcome',
                                                            common_causes)
            seq_ates.append((seq_arr, new_ate))


            new_distance = abs(new_ate - target_ate)
            if new_distance < smallest_distance_from_target:
                smallest_distance_from_target = new_distance
                distances_at_time_from_target.append((new_distance, time.time() - start_time))
            # if new_ate > largest_ate:
            #     largest_ate = new_ate
            #     largest_ate_and_time_list.append((largest_ate.item(), time.time() - start_time))
            #     print(f"smallest lists: {smallest_ate_and_time_list}",flush=True)
            #     print(f"largest lists: {largest_ate_and_time_list}", flush=True)
            # if new_ate < smallest_ate:
            #     smallest_ate = new_ate
            #     smallest_ate_and_time_list.append((smallest_ate.item(), time.time() - start_time))
            #     print(f"smallest lists: {smallest_ate_and_time_list}",flush=True)
            #     print(f"largest lists: {largest_ate_and_time_list}", flush=True)

            # if len(seq_arr) > 2 and Q_poped_num % 5 == 0:
                # for bin in bin_sequences(seq_ates, 300):
                #     bin_range = bin['range']
                #     bin_items = bin['items']
                #     if len(bin_items) > 0:
                #         print("=" * 40)
                #         print(f"range: {bin_range}")
                #         print(f"number of items: {len(bin_items)}")
                #         print(
                #             f"smallest seq in bin: {[] if len(bin_items) == 0 else sorted(bin_items, key=lambda x: len(x[0]))[0]}")

                # print(find_interesting(seq_ates,2))
                # print("-" * 120)
            # if Q_poped_num % 10000 ==0:
            #     print(f"all ates: {sorted(seq_ates, key=lambda x: x[1])}")
            #     print("="*50)

            if abs(new_ate - target_ate) < epsilon:
                print(
                    f"""***\nFINISHED\nATE before: {base_line_ate}\nATE now is: {new_ate}\nsequence is: {seq_arr}\n***""",
                    flush=True)
                solution_seq = seq_arr
                break

            if len(seq_arr) < max_seq_length:
                for func_name, col, move_bit in fast_moves:
                    try_count += 1
                    if mask & move_bit or (func_name.startswith("fill_") and curr_df[col].isna().sum() == 0):
                        prune_count += 1
                        continue
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
                        seen_dfs.add(df_new_signature)
                        new_mask = mask | move_bit
                        new_path = seq_arr + ((func_name, col),)

                        Q.append((new_path, new_mask))
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
        print(
            f"run time per popped from Q: mean: {np.mean(run_times_pop)}, percentiles={np.percentile(run_times_pop, [25, 75, 90, 95, 99]).tolist()}",
            flush=True)
        print(f"checked {try_count} combinations", flush=True)
        print(
            f"run time per neighbor: mean: {np.mean(run_times)}, percentiles={np.percentile(run_times, [25, 75, 90, 95, 99]).tolist()}",
            flush=True)
        print(f"distances from ATE (with time):\n{distances_at_time_from_target}", flush=True)
        # print(f"all ates: {sorted(seq_ates, key=lambda x: len(x[0]))}", flush=True)
        # print(f"all ates: {sorted(seq_ates, key=lambda x: x[1])}", flush=True)
        return solution_seq

    # def parallel_search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
    #                     max_seq_length: int):
    #     df_ = df.copy()
    #
    #     base_line_ate = get_base_line(common_causes, df_)
    #     Q = deque([[]])
    #     BATCH_SIZE = 30
    #     seen_dfs = set()
    #     Q_poped_num = 0
    #     prune_cnt = 0
    #     solution_seq = None
    #     seq_ates = []
    #     run_times_pop = []
    #     fast_dowhy_ate = FastDoWhyATE(df_, 'treatment', 'outcome', common_causes)
    #
    #     start_time = time.time()
    #     with ProcessPoolExecutor(max_workers=8) as pool:
    #         while len(Q) > 0:
    #             start_pop_Q_time = time.time()
    #             Q_poped_num += 1
    #             seq_arr = Q.popleft()
    #             curr_df = apply_data_preparations_seq(df_, seq_arr)
    #             curr_df_filled = curr_df.fillna(value=curr_df.mean())
    #             model = CausalModel(data=curr_df_filled, treatment='treatment', outcome='outcome',
    #                                 common_causes=common_causes)
    #             new_ate = fast_dowhy_ate.calculate_ate(model)
    #             seq_ates.append((seq_arr, new_ate))
    #
    #             # print(f"new ate is {new_ate}", flush=True)
    #             if abs(new_ate - target_ate) < epsilon:
    #                 print(
    #                     f"""***\nFINISHED\nATE before: {base_line_ate}\nATE now is: {new_ate}\nsequence is: {seq_arr}\n***""",
    #                     flush=True)
    #                 solution_seq = seq_arr
    #                 break
    #
    #             if len(seq_arr) < max_seq_length:
    #                 tasks = []
    #                 for col in common_causes:
    #                     for func_name, func in get_transformations().items():
    #                         tasks.append((seq_arr, curr_df, col, func_name, func, common_causes, seen_dfs))
    #
    #                 futures = [
    #                     pool.submit(self._process_col_func_batch, batch)
    #                     for batch in self.batched(tasks, BATCH_SIZE)
    #                 ]
    #                 for f in futures:  # as_completed(futures):
    #                     for res in f.result():
    #                         if res is None:
    #                             prune_cnt += 1
    #                             continue
    #                         new_seq_arr, df_sig = res
    #                         if df_sig not in seen_dfs:
    #                             seen_dfs.add(df_sig)
    #                             Q.append(new_seq_arr)
    #                         else:
    #                             prune_cnt += 1
    #
    #             end_pop_Q_time = time.time()
    #             run_times_pop.append(end_pop_Q_time - start_pop_Q_time)
    #     end_time = time.time()
    #     execution_time = end_time - start_time
    #
    #     print(f"Execution time: {execution_time} seconds", flush=True)
    #     print(f"pruned {prune_cnt}", flush=True)
    #     print(f"popped from Q {Q_poped_num} nodes", flush=True)
    #     print(
    #         f"run time per popped from Q: mean: {np.mean(run_times_pop)}, percentiles={np.percentile(run_times_pop, [25, 75, 90, 95, 99]).tolist()}",
    #         flush=True)
    #     # print(f"all ates: {sorted(seq_ates, key=lambda x: len(x[0]))}", flush=True)
    #
    #     return solution_seq
    #
    # def _process_col_func(self, args):
    #     seq_arr, df_, col, func_name, func, common_causes, seen_dfs = args
    #     # Skip already applied transformations
    #     if func_name in [f_name for (f_name, c) in seq_arr if c == col]:
    #         return None
    #
    #     # Apply the transformation
    #     new_df = df_.copy()
    #     new_df[col] = func(new_df[col])
    #     # df_sig = df_signature_fast(new_df, common_causes)
    #     df_sig = df_signature_fast_rounds(new_df, common_causes)
    #     # if df_sig in seen_dfs:
    #     #     return None
    #
    #     return (seq_arr + [(func_name, col)], df_sig)
    #
    # def batched(self, iterable, n):
    #     it = iter(iterable)
    #     while True:
    #         batch = list(islice(it, n))
    #         if not batch:
    #             break
    #         yield batch
    #
    # def _process_col_func_batch(self, batch):
    #     results = []
    #     for t in batch:
    #         results.append(self._process_col_func(t))
    #     return results
    #
    # ################################################################################
    #
    # def _evaluate_sequence(
    #         self, seq_arr: List[Tuple[str, str]],
    #         df_: pd.DataFrame,
    #         common_causes: List[str],
    #         target_ate: float,
    #         epsilon: float,
    #         max_seq_length: int,
    #         fast_dowhy_ate,
    #         seen_dfs: set
    # ) -> Tuple[
    #     List[Tuple[str, str]],  # solution sequence, or None
    #     List[Tuple[List[Tuple[str, str]], float]],  # results (seq, ate)
    #     List[List[Tuple[str, str]]],  # new sequences to add to Q
    #     int,  # prune count
    #     int,  # try count
    #     set  # new seen_dfs entries
    # ]:
    #     # Initialize objects needed within the process
    #     # Note: Objects like FastDoWhyATE must be re-initialized if they are not pickleable,
    #     # or if they hold large state that should not be copied to every process.
    #     # Passing the initialization arguments is often safer/more explicit.
    #     # fast_dowhy_ate = FastDoWhyATE(*fast_dowhy_ate_init_args)
    #
    #     prune_count = 0
    #     try_count = 0
    #     solution_seq = None
    #     seq_ates = []
    #     new_sequences = []
    #     new_seen_dfs_entries = set()
    #
    #     # 1. Apply Sequence and Calculate ATE
    #     curr_df = apply_data_preparations_seq(df_, seq_arr)
    #     curr_df_filled = curr_df.fillna(value=curr_df.mean())
    #     model = CausalModel(data=curr_df_filled, treatment='treatment', outcome='outcome',
    #                         common_causes=common_causes)
    #
    #     # Calculate ATE (This is the costly step we want to parallelize)
    #     new_ate = fast_dowhy_ate.calculate_ate(model)
    #     seq_ates.append((seq_arr, new_ate))
    #
    #     # 2. Check for Solution
    #     if abs(new_ate - target_ate) < epsilon:
    #         solution_seq = seq_arr
    #         # No need to generate new sequences if a solution is found
    #
    #     # 3. Generate Next Frontier Sequences (if not solution and not max depth)
    #     elif len(seq_arr) < max_seq_length:
    #         for col in common_causes:
    #             for func_name, func in get_transformations().items():
    #                 is_used = any(f_name == func_name and c == col for (f_name, c) in seq_arr)
    #                 is_fill_needed = func_name.startswith("fill_") and curr_df[col].isna().sum() > 0
    #
    #                 if not is_used or is_fill_needed:
    #                     try_count += 1
    #                     new_df = curr_df.copy()
    #                     new_df[col] = func(new_df[col])
    #
    #                     # Pruning: Check for no change in column
    #                     if new_df[col].equals(curr_df[col]):
    #                         prune_count += 1
    #                         continue
    #
    #                     df_new_signature = df_signature_fast(new_df, common_causes)
    #
    #                     # Pruning: Check if new signature is ALREADY in the global seen_dfs
    #                     # Note: We must check against the GLOBAL set to maintain BFS integrity
    #                     if df_new_signature in seen_dfs:
    #                         prune_count += 1
    #
    #                     # If df hasn't been globally explored:
    #                     else:
    #                         # Add new sequence and signature
    #                         new_sequences.append(seq_arr + [(func_name, col)])
    #                         new_seen_dfs_entries.add(df_new_signature)
    #                 else:
    #                     prune_count += 1
    #
    #     return solution_seq, seq_ates, new_sequences, prune_count, try_count, new_seen_dfs_entries
    #
    # def search_parallel_f(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
    #                       max_seq_length: int):
    #     df_ = df.copy()
    #     num_processes = mp.cpu_count()  # Use all available cores
    #     print('num processes', num_processes)
    #     base_line_ate = get_base_line(common_causes, df_)
    #     Q = deque([[]])  # Stores sequences to be processed in the next level
    #     seen_dfs = set()  # Global set for all explored dataframes
    #     prune_count = 0
    #     try_count = 0
    #     solution_seq = None
    #     all_seq_ates = []
    #     Q_poped_num = 0
    #
    #     # Arguments to re-initialize FastDoWhyATE in the worker processes
    #     fast_dowhy_ate = FastDoWhyATE(df_, 'treatment', 'outcome', common_causes)
    #     start_time = time.time()
    #
    #     # Use a Pool for parallel processing
    #     with mp.Pool(processes=num_processes) as pool:
    #
    #         while len(Q) > 0 and solution_seq is None:
    #
    #             # 1. Extract Current Frontier
    #             f_s_time = time.time()
    #             current_frontier = list(Q)
    #             Q.clear()  # Clear Q for the next level's sequences
    #             Q_poped_num += len(current_frontier)
    #
    #             # 2. Prepare arguments for parallel execution
    #             # We need to pass all necessary data to the helper function
    #             tasks = [
    #                 (seq_arr, df_, common_causes, target_ate, epsilon, max_seq_length, fast_dowhy_ate,
    #                  seen_dfs)
    #                 for seq_arr in current_frontier
    #             ]
    #
    #             # 3. Parallel Execution using pool.starmap
    #             # starmap applies the arguments from the tasks list to the function
    #             results = pool.starmap(self._evaluate_sequence, tasks)
    #
    #             # 4. Process Results
    #             for sol, seq_ates, new_sequences, p_count, t_count, new_seen_entries in results:
    #                 all_seq_ates.extend(seq_ates)
    #                 prune_count += p_count
    #                 try_count += t_count
    #
    #                 # Check for solution from any worker
    #                 if sol is not None:
    #                     solution_seq = sol
    #                     break  # Exit the loop over results
    #
    #                 # Merge new_seen_entries into the global set and add new sequences to Q
    #                 for entry in new_seen_entries:
    #                     # Critical check: only add if truly new to the GLOBAL set
    #                     if entry not in seen_dfs:
    #                         seen_dfs.add(entry)
    #
    #                 for seq in new_sequences:
    #                     # Re-check the signature of the sequence's resulting DF
    #                     # against the updated global seen_dfs before adding to Q.
    #                     # This avoids race conditions where two workers generate the same DF signature.
    #                     # Note: This is an approximation. A more robust solution might use a Manager dict for seen_dfs.
    #                     # However, for simple BFS, relying on the 'global' update post-`starmap` is often acceptable.
    #
    #                     # In this simple implementation, we rely on the check within _evaluate_sequence
    #                     # against the state of seen_dfs *at the start* of the level.
    #                     # The subsequent check is handled by the `new_seen_entries` merger logic above.
    #
    #                     # We only need to add the sequence to Q if its signature was in new_seen_entries
    #                     # and successfully added to the global seen_dfs.
    #                     Q.append(seq)
    #
    #             print(f"done frontier in {time.time() - f_s_time} sec. had {len(current_frontier)} nodes")
    #             # If a solution was found, the inner loop broke, so break the while loop
    #             if solution_seq is not None:
    #                 break
    #
    #     end_time = time.time()
    #     execution_time = end_time - start_time
    #
    #     # --- Output and Reporting (Keep your original reporting) ---
    #     if solution_seq is not None:
    #         final_ate = [ate for seq, ate in all_seq_ates if seq == solution_seq][0]
    #         print(
    #             f"""***\nFINISHED\nATE before: {base_line_ate}\nATE now is: {final_ate}\nsequence is: {solution_seq}\n***""",
    #             flush=True)
    #     else:
    #         print("Search finished without finding a solution within the given constraints.", flush=True)
    #
    #     print(f"Execution time: {execution_time} seconds (using {num_processes} processes)", flush=True)
    #     print(f"pruned {prune_count}", flush=True)
    #     print(f"popped from Q {Q_poped_num} nodes", flush=True)
    #     print(f"checked {try_count} combinations", flush=True)
    #     print(f"all ates: {sorted(all_seq_ates, key=lambda x: len(x[0]))}", flush=True)
    #
    #     return solution_seq
