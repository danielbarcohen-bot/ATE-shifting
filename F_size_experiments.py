import random
import time
from typing import List, Dict, Tuple

import pandas as pd

from search_methods.ATE_search import ATESearch


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
                 target_ate: float, epsilon: float,
                 transformations_dict: Dict, solution_sequence: Tuple,
                 whole_df_ops: List[str] = None, time_out_sec: int = 14400):
        """
        Args:
            search_algorithm: The ProbeATESearch instance
            df: DataFrame to search on
            common_causes: List of confounder columns
            target_ate: Target ATE value
            epsilon: Epsilon threshold for success
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
                F.append(f"{op}#TABLE")
        return F

    def _extract_F_solution(self) -> set:
        """Extract F elements used in solution sequence"""
        F_sol = set()
        for op, col in self.solution_sequence:
            if op in self.whole_df_ops:
                F_sol.add(f"{op}#TABLE")
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
        # if seed is not None:
        #     random.seed(seed)
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
        # subset_transformations = {op: self.transformations_dict[op] for op in operations_in_subset}
        subset_transformations = {op: self.transformations_dict[op]
                                  for op in self.transformations_dict.keys()
                                  if op in operations_in_subset}
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
            transformations_dict=subset_transformations,
            time_out_sec=self.time_out_sec,
            F_elements=F_subset,
            whole_df_ops=self.whole_df_ops
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
                     common_causes: List[str], target_ate: float, epsilon: float, transformations_dict: Dict,
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

    common_causes_twins = df_twins_no_missing_values.columns.difference(["treatment", "outcome"], sort=False).tolist()

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
