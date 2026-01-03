# import heapq
# import time
# from typing import List
#
# import pandas as pd
# from dowhy import CausalModel
#
# from fast_dowhy_ATE import FastDoWhyATE
# from search_methods.ATE_search import ATESearch
# from utils import df_signature_fast, apply_data_preparations_seq, get_transformations
# from concurrent.futures import ProcessPoolExecutor, as_completed
#
#
#
# class Node:
#     def __init__(self, g: int, h: float, sequence: List[str], common_causes: List[str], ATE: float):
#         # self.df = df
#         self.g = g  # cost so far (# operations)
#         self.h = h  # heuristic
#         self.f = g + h  # priority
#         self.sequence = sequence
#         self.common_causes = common_causes
#         self.ATE = ATE
#
#     def __lt__(self, other):
#         return self.f < other.f
#
#
# def heuristic_func(target_ate, epsilon, current_ate):
#     return min(abs(target_ate - epsilon - current_ate), abs(target_ate + epsilon - current_ate))
#
#
# # def expand_node(func, func_name, curr_df, col, common_causes, fast_dowhy_ate, curr_g, target_ate, epsilon, sequence):
# def expand_node(batch, curr_df, common_causes, fast_dowhy_ate, target_ate, epsilon, curr_g, sequence):
#     nodes_and_hashes = []
#     base_filled_df = curr_df.fillna(value=curr_df.mean())
#     for func, func_name, col in batch:
#         new_col = func(curr_df[col].copy())
#         new_df = curr_df.copy(deep=False)
#         new_df[col] = new_col
#         new_hash = df_signature_fast(new_df, common_causes)
#         base_filled_df_copy = base_filled_df.copy(deep=False)
#         base_filled_df_copy[col] = new_col
#         base_filled_df_copy[col].fillna(new_col.mean(), inplace=True)
#         model = CausalModel(data=base_filled_df_copy, treatment='treatment', outcome='outcome',
#                             common_causes=common_causes)
#         new_ate = fast_dowhy_ate.calculate_ate(model)
#
#         g_new = curr_g + 1
#         h_new = heuristic_func(target_ate, epsilon, new_ate)
#         new_node = Node(g=g_new, h=h_new, sequence=sequence + [(func_name, col)],
#                         common_causes=common_causes, ATE=new_ate)
#         nodes_and_hashes.append((new_node, new_hash))
#
#     return nodes_and_hashes
#
# def chunk_tasks(tasks, batch_size):
#     for i in range(0, len(tasks), batch_size):
#         yield tasks[i:i+batch_size]
#
# class AStarATESearch(ATESearch):
#     # def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
#     #            max_seq_length: int):
#     #     start_time = time.time()
#     #     df_ = df.copy()
#     #     fast_dowhy_ate = FastDoWhyATE(df_, 'treatment', 'outcome', common_causes)
#     #
#     #     curr_df_filled = df_.fillna(value=df_.mean())
#     #     model = CausalModel(data=curr_df_filled, treatment='treatment', outcome='outcome',
#     #                         common_causes=common_causes)
#     #     curr_ate = fast_dowhy_ate.calculate_ate(model)
#     #     h0 = heuristic_func(target_ate, epsilon, curr_ate)
#     #
#     #     start_node = Node(g=0, h=h0, sequence=[], common_causes=common_causes, ATE=curr_ate)
#     #
#     #     open_heap = []
#     #     heapq.heappush(open_heap, start_node)
#     #     visited = {df_signature_fast(df_, common_causes): 0}  # state hash -> best g encountered
#     #
#     #     it = 0
#     #     L, U = target_ate - epsilon, target_ate + epsilon
#     #     while open_heap:  # and it < max_seq_length:
#     #         it += 1
#     #         node = heapq.heappop(open_heap)
#     #
#     #         # Check goal condition
#     #         curr_df = apply_data_preparations_seq(df_, node.sequence)
#     #         if L <= node.ATE <= U:
#     #             print(f"Reached goal after {it} expansions.")
#     #             print(f"sequence: {node.sequence}")
#     #             print(f"ATE is {node.ATE}")
#     #             print("took: ", time.time() - start_time)
#     #             return node.sequence, node.ATE
#     #
#     #         # Expand operations
#     #         if len(node.sequence) < max_seq_length:
#     #             nigger_time = time.time()
#     #             for col in common_causes:
#     #                 for func_name, func in get_transformations().items():
#     #                     if not (func_name.startswith("fill_") and curr_df[col].isna().sum() > 0):
#     #                         expand_node_start_time = time.time()
#     #                         new_col = func(curr_df[col].copy())
#     #
#     #                         # canonical_sequence = canonical(seq_arr + [(func_name, col)])
#     #                         # if canonical_sequence in seen_seq or new_col.equals(
#     #                         #         curr_df[col]):  # col hasn't changed - so same df!
#     #                         #     prune_count += 1
#     #                         #     continue
#     #                         # seen_seq.add(canonical_sequence)
#     #                         new_df = curr_df.copy(deep=False)
#     #                         new_df[col] = new_col
#     #                         new_hash = df_signature_fast(new_df, common_causes)
#     #
#     #                         new_df_filled = new_df.fillna(value=new_df.mean())
#     #                         model = CausalModel(data=new_df_filled, treatment='treatment', outcome='outcome',
#     #                                             common_causes=common_causes)
#     #                         new_ate = fast_dowhy_ate.calculate_ate(model)
#     #
#     #                         g_new = node.g + 1
#     #                         if new_hash in visited and visited[new_hash] <= g_new:
#     #                             continue  # already reached with cheaper cost
#     #
#     #                         h_new = heuristic_func(target_ate, epsilon, new_ate)
#     #                         new_node = Node(g=g_new, h=h_new, sequence=node.sequence + [(func_name, col)],
#     #                                         common_causes=common_causes, ATE=new_ate)
#     #
#     #                         visited[new_hash] = g_new
#     #                         heapq.heappush(open_heap, new_node)
#     #             print(f"expand node all son took {time.time() - nigger_time}")
#     #         # # Scale operations with parameters
#     #         # for k in SCALE_FACTORS:
#     #         #     new_df = op_scale(node.df, target_col, k=k)
#     #         #     new_hash = state_hash(new_df)
#     #         #     g_new = node.g + 1
#     #         #     if new_hash in visited and visited[new_hash] <= g_new:
#     #         #         continue
#     #         #
#     #         #     h_new = heuristic_func(new_df, target_col, y_col, controls, beta_goal)
#     #         #     new_node = Node(new_df, g=g_new, h=h_new,
#     #         #                     path=node.path + [f"scale({k})"])
#     #         #
#     #         #     visited[new_hash] = g_new
#     #         #     heapq.heappush(open_heap, new_node)
#     #
#     #     print("A* failed to find solution within iteration limit.")
#     #     print("took: ", time.time() - start_time)
#     #     return None, None, None
#
#
#     def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float,
#                max_seq_length: int):
#         print("start A*")
#         batch_size = 8
#         start_time = time.time()
#         df_ = df.copy()
#         fast_dowhy_ate = FastDoWhyATE(df_, 'treatment', 'outcome', common_causes)
#
#         curr_df_filled = df_.fillna(value=df_.mean())
#         model = CausalModel(data=curr_df_filled, treatment='treatment', outcome='outcome',
#                             common_causes=common_causes)
#         curr_ate = fast_dowhy_ate.calculate_ate(model)
#         h0 = heuristic_func(target_ate, epsilon, curr_ate)
#
#         start_node = Node(g=0, h=h0, sequence=[], common_causes=common_causes, ATE=curr_ate)
#
#         open_heap = []
#         heapq.heappush(open_heap, start_node)
#         visited = {df_signature_fast(df_, common_causes): 0}  # state hash -> best g encountered
#
#         it = 0
#         L, U = target_ate - epsilon, target_ate + epsilon
#         with ProcessPoolExecutor(max_workers=16) as pool:
#             while open_heap:  # and it < max_seq_length:
#                 it += 1
#                 node = heapq.heappop(open_heap)
#
#                 # Check goal condition
#                 curr_df = apply_data_preparations_seq(df_, node.sequence)
#                 if L <= node.ATE <= U:
#                     print(f"Reached goal after {it} expansions.")
#                     print(f"sequence: {node.sequence}")
#                     print(f"ATE is {node.ATE}")
#                     print("took: ", time.time() - start_time)
#                     return node.sequence, node.ATE
#
#                 # Expand operations
#                 if len(node.sequence) < max_seq_length:
#                     expand_start_time = time.time()
#                     tasks = []
#
#                     for col in common_causes:
#                         for func_name, func in get_transformations().items():
#                             if not (func_name.startswith("fill_") and curr_df[col].isna().sum() > 0):
#                                 tasks.append((func, func_name, col))
#                                 # tasks.append(pool.submit(expand_node,  func, func_name, curr_df, col, common_causes, fast_dowhy_ate, node.g, target_ate, epsilon, node.sequence))
#                     futures = [pool.submit(expand_node, batch, curr_df, common_causes, fast_dowhy_ate, target_ate, epsilon, node.g, node.sequence) for batch in list(chunk_tasks(tasks, batch_size))]
#                     s = time.time()
#                     for future in as_completed(futures):
#                         for new_node, new_hash in future.result():
#                             if new_hash in visited and visited[new_hash] <= new_node.g:
#                                 continue
#                             visited[new_hash] = new_node.g
#                             heapq.heappush(open_heap, new_node)
#                     print(f"tasks to do: {len(tasks)}")
#                     print(f"expand node took: {time.time() - expand_start_time}")
#                     print(f"run on all completed took: {time.time() - s}")
#
#         print("A* failed to find solution within iteration limit.")
#         print("took: ", time.time() - start_time)
#         return None, None, None
