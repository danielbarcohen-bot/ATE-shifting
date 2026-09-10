import random
from collections import defaultdict
from itertools import product
from typing import Callable, List

import pandas as pd

from utils import apply_data_preparations_seq, calculate_ate_linear_regression_lstsq


class RandomSearch:
    def search(self, df: pd.DataFrame, common_causes: List[str],transformations_dict: dict[str, Callable], sequence_length: int, legal_ops_by_type=None):
        col_types = df.attrs.get('col_types', None)
        actions = [(col, transformation) for col, transformation in product(common_causes, transformations_dict.keys())]
        seen_transformation_classes_per_col = defaultdict(list)
        sequence = ()
        for _ in range(sequence_length):
            found_action = False
            while not found_action:
                col, transformation = random.choice(actions)
                if transformation == "isolationForest" and any(f_n == "isolationForest" for f_n, c in sequence):
                    continue
                if transformation == "drop_duplicates" and any(f_n == "drop_duplicates" for f_n, c in sequence):
                    continue
                col_type = col_types.get(col) if col_types else None
                if col_type is not None and transformation not in legal_ops_by_type.get(col_type, []):
                    continue  # illegal combo (e.g. normalize on a binary col) — skip
                trans_class = transformation.split("_")[0]
                if trans_class not in seen_transformation_classes_per_col[col]:
                    seen_transformation_classes_per_col[col].append(trans_class)
                    sequence = sequence + ((transformation, col),)
                    found_action = True
        curr_df = apply_data_preparations_seq(df, sequence, transformations_dict)
        new_ate = calculate_ate_linear_regression_lstsq(curr_df, 'treatment', 'outcome', common_causes)
        return sequence, new_ate


