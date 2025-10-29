from typing import List
import pandas as pd

from search_methods.brute_force_ATE_search import BruteForceATESearch
from search_methods.pruning_ATE_search import PruneATESearch


class Experiment:
    def __init__(
            self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float = 0.0001,
            max_length: int = 5) -> None:
        self.df = df
        self.common_causes = common_causes
        self.target_ate = target_ate
        self.epsilon = epsilon
        self.max_length = max_length

    def run_brute(self):
        return BruteForceATESearch().search(df=self.df, common_causes=self.common_causes, target_ate=self.target_ate,
                                            epsilon=self.epsilon,
                                            max_seq_length=self.max_length)

    def run_prune(self):
        return PruneATESearch().search(df=self.df, common_causes=self.common_causes, target_ate=self.target_ate,
                                       epsilon=self.epsilon,
                                       max_seq_length=self.max_length)
