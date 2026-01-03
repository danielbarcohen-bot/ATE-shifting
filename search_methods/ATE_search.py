from abc import ABC, abstractmethod
from typing import List, Callable

import pandas as pd


# from utils import analyze_ate_risk, find_droppable_features_by_budget


class ATESearch(ABC):

    @abstractmethod
    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float, max_seq_length: int,
               transformations_dict: dict[str, Callable]):
        pass
