# import math
# import random
# from typing import List, Dict, Tuple
# import pandas as pd
# import time
#
# from search_methods.ATE_search import ATESearch
# from utils import apply_data_preparations_seq, calculate_ate_linear_regression_lstsq, \
#     get_base_line, df_signature_fast, calculate_ate_with_uncertainty
#
#
# class OperationSpaceExperiment:
#     """
#     Experiment to test search performance across different subsets of operation space.
#
#     Process:
#     1. G = F \ F_solution (all elements except solution)
#     2. Split G into 15 subgroups
#     3. For i in range(1, |G|/15 + 1):
#        - Take i subgroups, combine them, add F_solution
#        - Run search algorithm on this subset
#     4. Repeat 3 times with different random shuffles of G
#     """
#
#     def __init__(self, search_algorithm: ATESearch, df: pd.DataFrame, common_causes: List[str],
#                  target_ate: float, epsilon: float, max_seq_length: int,
#                  transformations_dict: Dict, solution_sequence: Tuple,
#                  whole_df_ops: List[str] = None, time_out_sec: int = 14400):
#         """
#         Args:
#             search_algorithm: The ProbeATESearch instance
#             df: DataFrame to search on
#             common_causes: List of confounder columns
#             target_ate: Target ATE value
#             epsilon: Epsilon threshold for success
#             max_seq_length: Max sequence length
#             transformations_dict: Dict of all available operations
#             solution_sequence: The known solution sequence [(op, col), (op, col), ...]
#             whole_df_ops: List of operation names that apply to whole df (e.g., ['isolationForest'])
#             time_out_sec: Timeout in seconds
#         """
#         self.search_algorithm = search_algorithm
#         self.df = df
#         self.common_causes = common_causes
#         self.target_ate = target_ate
#         self.epsilon = epsilon
#         self.max_seq_length = max_seq_length
#         self.transformations_dict = transformations_dict
#         self.solution_sequence = solution_sequence
#         self.time_out_sec = time_out_sec
#         self.whole_df_ops = whole_df_ops or []
#
#         # F = all operation-column pairs + whole-df operations
#         self.F_all = self._build_F()
#         self.F_solution = self._extract_F_solution()
#         # G = F \ F_solution
#         self.G = self._get_G()
#
#     def _build_F(self) -> List[str]:
#         """Build F: column-specific pairs like {norm#x, norm#y, ...} + whole-df ops {isolationForest}"""
#         F = []
#         # Add column-specific operations
#         for op in self.transformations_dict.keys():
#             if op not in self.whole_df_ops:
#                 for col in self.common_causes:
#                     F.append(f"{op}#{col}")
#         # Add whole-df operations (appear once, no column suffix)
#         for op in self.whole_df_ops:
#             if op in self.transformations_dict:
#                 F.append(op)
#         return F
#
#     def _extract_F_solution(self) -> set:
#         """Extract F elements used in solution sequence"""
#         F_sol = set()
#         for op, col in self.solution_sequence:
#             if op in self.whole_df_ops:
#                 F_sol.add(op)  # No column suffix for whole-df ops
#             else:
#                 F_sol.add(f"{op}#{col}")
#         return F_sol
#
#     def _get_G(self) -> List[str]:
#         """Get G = F \ F_solution (all F elements except those in solution)"""
#         return [f_elem for f_elem in self.F_all if f_elem not in self.F_solution]
#
#     def split_G_into_15_groups(self, G: List[str]) -> List[List[str]]:
#         """
#         Split G into 15 subgroups as evenly as possible.
#
#         Args:
#             G: List of non-solution F elements
#
#         Returns:
#             List of 15 groups
#         """
#         num_groups = 15
#         if len(G) < num_groups:
#             raise ValueError(f"G size ({len(G)}) cannot be less than 15")
#
#         shuffled_G = G.copy()
#         random.shuffle(shuffled_G)
#
#         groups = [[] for _ in range(num_groups)]
#         for idx, f_elem in enumerate(shuffled_G):
#             groups[idx % num_groups].append(f_elem)
#
#         return groups
#
#     def run_experiment(self, num_trials: int = 3, seed: int = None) -> Dict:
#         """
#         Run the full experiment.
#
#         For each of num_trials:
#             1. Split G into 15 subgroups
#             2. For i in range(1, |G|/15 + 1):
#                - Take first i subgroups, combine and add F_solution
#                - Run search on this subset
#
#         Args:
#             num_trials: Number of trials with different G shuffles
#             seed: Random seed for reproducibility
#
#         Returns:
#             Dict with results for each trial and i value
#         """
#         if seed is not None:
#             random.seed(seed)
#
#         num_groups = 15
#         max_i = num_groups  # Always iterate from 1 to 15
#
#         results = {
#             'config': {
#                 'total_F_elements': len(self.F_all),
#                 'F_solution': list(self.F_solution),
#                 'G_size': len(self.G),
#                 'num_subgroups': num_groups,
#                 'max_i': max_i,
#                 'i_values': list(range(1, max_i + 1)),
#                 'num_trials': num_trials,
#                 'epsilon': self.epsilon,
#                 'target_ate': self.target_ate
#             },
#             'trials': []
#         }
#
#         for trial_num in range(num_trials):
#             print(f"\n{'='*70}")
#             print(f"TRIAL {trial_num + 1}/{num_trials}")
#             print(f"{'='*70}")
#
#             # Split G into 15 groups with fresh shuffle
#             groups = self.split_G_into_15_groups(self.G)
#
#             print(f"G split into 15 groups:")
#             for g_idx, group in enumerate(groups):
#                 print(f"  Group {g_idx}: {len(group)} elements")
#
#             trial_results = {
#                 'trial_num': trial_num + 1,
#                 'groups': groups,
#                 'i_results': {}
#             }
#
#             # For each i from 1 to max_i
#             for i in range(1, max_i + 1):
#                 print(f"\n  i={i} (combining first {i} subgroups)")
#
#                 # Combine first i subgroups
#                 combined_G = []
#                 for group_idx in range(i):
#                     combined_G.extend(groups[group_idx])
#
#                 # Add F_solution
#                 subset_F = combined_G + list(self.F_solution)
#
#                 print(f"    |subset F| = {len(combined_G)} (from G) + {len(self.F_solution)} (solution) = {len(subset_F)}")
#
#                 # Run search on this subset
#                 subset_results = self._run_search_on_subset(subset_F)
#                 trial_results['i_results'][i] = subset_results
#
#                 # Print summary
#                 status = "✓ FOUND" if subset_results['found_solution'] else "✗ NOT FOUND"
#                 print(f"    Result: {status}", end="")
#                 if subset_results['found_solution']:
#                     print(f" | Time: {subset_results['time_sec']:.2f}s | Checked: {subset_results['checked']}")
#                 else:
#                     print(f" | Timed out after {subset_results['time_sec']:.2f}s | Best error: {subset_results['best_error']:.6f}")
#
#             results['trials'].append(trial_results)
#
#         return results
#
#     def _run_search_on_subset(self, F_subset: List[str]) -> Dict:
#         """
#         Run search algorithm on a specific subset of F elements.
#
#         Args:
#             F_subset: List of F elements (op#col pairs or whole-df ops) to use
#
#         Returns:
#             Dict with search results
#         """
#         # Extract unique operations from F elements
#         operations_in_subset = set()
#         for f_elem in F_subset:
#             if "#" in f_elem:
#                 # Column-specific: op#col -> op
#                 op = f_elem.split("#")[0]
#                 operations_in_subset.add(op)
#             else:
#                 # Whole-df operation: just the op name
#                 operations_in_subset.add(f_elem)
#
#         # Create subset transformations_dict with only these operations
#         subset_transformations = {op: self.transformations_dict[op] for op in operations_in_subset}
#
#         # Wrap the search to capture results
#         result = {
#             'found_solution': False,
#             'time_sec': 0,
#             'final_ate': None,
#             'best_error': float('inf'),
#             'checked': 0,
#             'solution_sequence': None,
#             'distances_from_target': []
#         }
#
#         start_time = time.time()
#         try:
#             # Pass F_subset to the search so it uses the exact F elements we want
#             # NOTE: ProbeATESearch.search() signature needs to accept F_elements parameter
#             # and pass it to ProbManager(F_elements=F_subset)
#             self.search_algorithm.search(
#                 df=self.df,
#                 common_causes=self.common_causes,
#                 target_ate=self.target_ate,
#                 epsilon=self.epsilon,
#                 max_seq_length=self.max_seq_length,
#                 transformations_dict=subset_transformations,
#                 time_out_sec=self.time_out_sec,
#                 F_elements=F_subset  # NEW: pass exact F elements to search
#             )
#             # If search completes without exception, it found the solution
#             result['found_solution'] = True
#         except SystemExit:
#             # Search found solution and called exit()
#             result['found_solution'] = True
#         except Exception as e:
#             print(f"    Error during search: {e}")
#         finally:
#             result['time_sec'] = time.time() - start_time
#
#         return result
#
#
# def run_full_experiment(search_algorithm: ATESearch, df: pd.DataFrame,
#                        common_causes: List[str], target_ate: float, epsilon: float,
#                        max_seq_length: int, transformations_dict: Dict,
#                        solution_sequence: Tuple,
#                        whole_df_ops: List[str] = None, num_trials: int = 3, seed: int = None) -> Dict:
#     """
#     Convenience function to run the experiment.
#
#     G = F \ F_solution
#     For each trial:
#         - Split G into 15 subgroups
#         - For i in 1..|G|/15:
#             - Combine first i subgroups, add F_solution
#             - Run search
#
#     Usage example:
#     ```
#     results = run_full_experiment(
#         search_algorithm=search_algo,
#         df=df,
#         common_causes=['X1', 'X2'],
#         target_ate=0.5,
#         epsilon=0.1,
#         max_seq_length=5,
#         transformations_dict=all_transformations,
#         solution_sequence=(('removeOutliers', 'treatment'), ('normalize', 'outcome')),
#         whole_df_ops=['isolationForest'],
#         num_trials=3,
#         seed=42
#     )
#     ```
#     """
#     experiment = OperationSpaceExperiment(
#         search_algorithm=search_algorithm,
#         df=df,
#         common_causes=common_causes,
#         target_ate=target_ate,
#         epsilon=epsilon,
#         max_seq_length=max_seq_length,
#         transformations_dict=transformations_dict,
#         solution_sequence=solution_sequence,
#         whole_df_ops=whole_df_ops,
#         time_out_sec=14400
#     )
#
#     results = experiment.run_experiment(num_trials=num_trials, seed=seed)
#     return results




#
# """
# Extract data from experiment results and create graphs.
# X-axis: F size (|i subgroups| + |F_solution|)
# Y-axis: Runtime
# """
#
# import json
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from typing import Dict, List, Tuple
#
#
# def extract_results_to_dataframe(results: Dict) -> pd.DataFrame:
#     """
#     Extract results into a DataFrame for easy plotting.
#
#     Args:
#         results: Output from run_full_experiment()
#
#     Returns:
#         DataFrame with columns: trial, i, F_size, time_sec, found_solution
#     """
#     config = results['config']
#     F_solution_size = len(config['F_solution'])
#     G_size = config['G_size']
#
#     rows = []
#
#     for trial in results['trials']:
#         trial_num = trial['trial_num']
#
#         for i, i_result in trial['i_results'].items():
#             # Calculate actual F size for this i
#             # i subgroups were sampled, each group has roughly G_size/15 elements
#             elements_from_G = i * (G_size / 15)
#             F_size = elements_from_G + F_solution_size
#
#             row = {
#                 'trial': trial_num,
#                 'i': i,
#                 'F_size': F_size,
#                 'time_sec': i_result['time_sec'],
#                 'found_solution': i_result['found_solution'],
#                 'best_error': i_result['best_error']
#             }
#             rows.append(row)
#
#     return pd.DataFrame(rows)
#
#
# def plot_runtime_vs_F_size(results: Dict, save_path: str = None):
#     """
#     Plot runtime vs F size with separate lines for each trial.
#
#     Args:
#         results: Output from run_full_experiment()
#         save_path: Optional path to save the figure
#     """
#     df = extract_results_to_dataframe(results)
#
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     # Plot each trial separately
#     for trial_num in sorted(df['trial'].unique()):
#         trial_data = df[df['trial'] == trial_num].sort_values('F_size')
#
#         # Only plot successful runs
#         successful = trial_data[trial_data['found_solution']]
#
#         ax.plot(successful['F_size'], successful['time_sec'],
#                 marker='o', label=f'Trial {trial_num}', linewidth=2, markersize=6)
#
#     ax.set_xlabel('F Size (number of operation-column pairs)', fontsize=12)
#     ax.set_ylabel('Runtime (seconds)', fontsize=12)
#     ax.set_title('Search Runtime vs Operation Space Size', fontsize=14)
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Figure saved to {save_path}")
#
#     plt.show()
#     return fig, ax

import math
import random
from typing import List, Dict, Tuple
import pandas as pd
import time

from search_methods.ATE_search import ATESearch
from utils import apply_data_preparations_seq, calculate_ate_linear_regression_lstsq, \
    get_base_line, df_signature_fast, calculate_ate_with_uncertainty


class OperationSpaceExperiment:
    """
    Experiment to test search performance across different subsets of operation space.

    Process:
    1. G = F \ F_solution (all elements except solution)
    2. Split G into 15 subgroups
    3. For i in range(1, |G|/15 + 1):
       - Take i subgroups, combine them, add F_solution
       - Run search algorithm on this subset
    4. Repeat 3 times with different random shuffles of G
    """

    def __init__(self, search_algorithm: ATESearch, df: pd.DataFrame, common_causes: List[str],
                 target_ate: float, epsilon: float, max_seq_length: int,
                 transformations_dict: Dict, solution_sequence: Tuple,
                 whole_df_ops: List[str] = None, time_out_sec: int = 14400):
        """
        Args:
            search_algorithm: The ProbeATESearch instance
            df: DataFrame to search on
            common_causes: List of confounder columns
            target_ate: Target ATE value
            epsilon: Epsilon threshold for success
            max_seq_length: Max sequence length
            transformations_dict: Dict of all available operations
            solution_sequence: The known solution sequence [(op, col), (op, col), ...]
            whole_df_ops: List of operation names that apply to whole df (e.g., ['isolationForest'])
            time_out_sec: Timeout in seconds
        """
        self.search_algorithm = search_algorithm
        self.df = df
        self.common_causes = common_causes
        self.target_ate = target_ate
        self.epsilon = epsilon
        self.max_seq_length = max_seq_length
        self.transformations_dict = transformations_dict
        self.solution_sequence = solution_sequence
        self.time_out_sec = time_out_sec
        self.whole_df_ops = whole_df_ops or []

        # F = all operation-column pairs + whole-df operations
        self.F_all = self._build_F()
        self.F_solution = self._extract_F_solution()
        # G = F \ F_solution
        self.G = self._get_G()

    def _build_F(self) -> List[str]:
        """Build F: column-specific pairs like {norm#x, norm#y, ...} + whole-df ops {isolationForest}"""
        F = []
        # Add column-specific operations
        for op in self.transformations_dict.keys():
            if op not in self.whole_df_ops:
                for col in self.common_causes:
                    F.append(f"{op}#{col}")
        # Add whole-df operations (appear once, no column suffix)
        for op in self.whole_df_ops:
            if op in self.transformations_dict:
                F.append(op)
        return F

    def _extract_F_solution(self) -> set:
        """Extract F elements used in solution sequence"""
        F_sol = set()
        for op, col in self.solution_sequence:
            if op in self.whole_df_ops:
                F_sol.add(op)  # No column suffix for whole-df ops
            else:
                F_sol.add(f"{op}#{col}")
        return F_sol

    def _get_G(self) -> List[str]:
        """Get G = F \ F_solution (all F elements except those in solution)"""
        return [f_elem for f_elem in self.F_all if f_elem not in self.F_solution]

    def split_G_into_15_groups(self, G: List[str]) -> List[List[str]]:
        """
        Split G into 15 subgroups as evenly as possible.

        Args:
            G: List of non-solution F elements

        Returns:
            List of 15 groups
        """
        num_groups = 15
        if len(G) < num_groups:
            raise ValueError(f"G size ({len(G)}) cannot be less than 15")

        shuffled_G = G.copy()
        random.shuffle(shuffled_G)

        groups = [[] for _ in range(num_groups)]
        for idx, f_elem in enumerate(shuffled_G):
            groups[idx % num_groups].append(f_elem)

        return groups


    def run_experiment(self, i: int, seed: int = None) -> Dict:
        """
        Run the experiment once for a selected i.
        """
        if seed is not None:
            random.seed(seed)
        num_groups = 15
        if not (1 <= i <= num_groups):
            raise ValueError(f"i must be between 1 and {num_groups}")
        groups = self.split_G_into_15_groups(self.G)
        combined_G = []
        for group_idx in range(i):
            combined_G.extend(groups[group_idx])
        subset_F = combined_G + list(self.F_solution)

        print(f"RUNNING EXPERIMENT WITH i={i}. G size is {len(combined_G)} + {len(self.F_solution)}")
        subset_results = self._run_search_on_subset(subset_F)
        return {
            "config": {
                "total_F_elements": len(self.F_all),
                "F_solution": list(self.F_solution),
                "G_size": len(self.G),
                "num_subgroups": num_groups,
                "selected_i": i,
                "epsilon": self.epsilon,
                "target_ate": self.target_ate,
            },
            "groups": groups,
            "subset_size": len(subset_F),
            "results": subset_results,
        }


    def _run_search_on_subset(self, F_subset: List[str]) -> Dict:
        """
        Run search algorithm on a specific subset of F elements.

        Args:
            F_subset: List of F elements (op#col pairs or whole-df ops) to use

        Returns:
            Dict with search results
        """
        # Extract unique operations from F elements
        operations_in_subset = set()
        for f_elem in F_subset:
            if "#" in f_elem:
                # Column-specific: op#col -> op
                op = f_elem.split("#")[0]
                operations_in_subset.add(op)
            else:
                # Whole-df operation: just the op name
                operations_in_subset.add(f_elem)

        # Create subset transformations_dict with only these operations
        subset_transformations = {op: self.transformations_dict[op] for op in operations_in_subset}

        # Wrap the search to capture results
        result = {
            'found_solution': False,
            'time_sec': 0,
            'final_ate': None,
            'best_error': float('inf'),
            'checked': 0,
            'solution_sequence': None,
            'distances_from_target': []
        }

        start_time = time.time()
        # try:
            # Pass F_subset to the search so it uses the exact F elements we want
            # NOTE: ProbeATESearch.search() signature needs to accept F_elements parameter
            # and pass it to ProbManager(F_elements=F_subset)
        self.search_algorithm.search(
            df=self.df,
            common_causes=self.common_causes,
            target_ate=self.target_ate,
            epsilon=self.epsilon,
            max_seq_length=self.max_seq_length,
            transformations_dict=subset_transformations,
            time_out_sec=self.time_out_sec,
            F_elements=F_subset  # NEW: pass exact F elements to search
        )
        # If search completes without exception, it found the solution
        result['found_solution'] = True
        # except SystemExit:
        #     # Search found solution and called exit()
        #     result['found_solution'] = True
        # except Exception as e:
        #     print(f"    Error during search: {e}")
        # finally:
        #     result['time_sec'] = time.time() - start_time

        return result


def run_F_experiment(search_algorithm: ATESearch, df: pd.DataFrame,
                        common_causes: List[str], target_ate: float, epsilon: float,
                        max_seq_length: int, transformations_dict: Dict,
                        solution_sequence: Tuple,
                        whole_df_ops: List[str] = None, i: int = 15, seed: int = None) -> Dict:
    """
    Convenience function to run the experiment.

    G = F \ F_solution
    For each trial:
        - Split G into 15 subgroups
        - For i in 1..|G|/15:
            - Combine first i subgroups, add F_solution
            - Run search

    Usage example:
    ```
    results = run_full_experiment(
        search_algorithm=search_algo,
        df=df,
        common_causes=['X1', 'X2'],
        target_ate=0.5,
        epsilon=0.1,
        max_seq_length=5,
        transformations_dict=all_transformations,
        solution_sequence=(('removeOutliers', 'treatment'), ('normalize', 'outcome')),
        whole_df_ops=['isolationForest'],
        num_trials=3,
        seed=42
    )
    ```
    """
    experiment = OperationSpaceExperiment(
        search_algorithm=search_algorithm,
        df=df,
        common_causes=common_causes,
        target_ate=target_ate,
        epsilon=epsilon,
        max_seq_length=max_seq_length,
        transformations_dict=transformations_dict,
        solution_sequence=solution_sequence,
        whole_df_ops=whole_df_ops,
        time_out_sec=14400
    )

    results = experiment.run_experiment(i=i, seed=seed)
    return results


if __name__ == "__main__":
    from search_methods.probe_ATE_search import ProbeATESearch
    from experiments import largest_data_transformations, df_twins_no_missing_values

    probe_search_alg = ProbeATESearch()

    common_causes_twins = df_twins_no_missing_values.columns.difference(["treatment", "outcome"]).tolist()

    results = run_F_experiment(
        search_algorithm=probe_search_alg,
        df=df_twins_no_missing_values,
        common_causes=common_causes_twins,
        target_ate=-0.06,
        epsilon=0.06,
        max_seq_length=5,
        transformations_dict=largest_data_transformations,
        solution_sequence=(('bin_equal_frequency_2', 'wt'), ('norm_log', 'gestat10')),
        i=15,
        whole_df_ops=['isolationForest'],
        seed=42  # For reproducibility
    )