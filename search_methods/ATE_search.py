from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class ATESearch(ABC):

    @abstractmethod
    def search(self, df: pd.DataFrame, common_causes: List[str], target_ate: float, epsilon: float, max_seq_length: int):
        pass
