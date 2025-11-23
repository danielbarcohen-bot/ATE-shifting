from abc import ABC, abstractmethod
from typing import List
import pandas as pd

from utils import analyze_ate_risk, find_droppable_features_by_budget


class ATESearch(ABC):

    @abstractmethod
    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float, max_seq_length: int):
        pass

    def drop_search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float, max_seq_length: int):  # drops some features with low influence
        df_ = df.copy()

        df_['treatment'] = df_['treatment'].astype(int)
        df_.fillna(df_.mean(), inplace=True)
        summary, details = analyze_ate_risk(df_, 'treatment', 'outcome')

        new_epsilon = epsilon / 2
        max_budget = epsilon - new_epsilon
        dropped_by_budget, actual_risk = find_droppable_features_by_budget(details, max_budget)

        return self.search(df.drop(columns=dropped_by_budget), common_causes, target_ate, new_epsilon, max_seq_length)
