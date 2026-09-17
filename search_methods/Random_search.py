import random
from collections import defaultdict
from itertools import product
from typing import Callable, List, Dict

import pandas as pd

from utils import apply_data_preparations_seq, calculate_ate_linear_regression_lstsq, get_fill_combinations


class RandomSearch:
    def search(self,
               df: pd.DataFrame,
               common_causes: List[str],
               transformations_dict: dict[str, Callable],
               whole_table_ops: List[str],
               sequence_length: int,
               legal_ops_by_type,
               sequence_to_extend = None
               ):
        col_types = df.attrs.get('col_types', None)
        actions = [
            (col, transformation)
            for col in common_causes
            for transformation in transformations_dict.keys()
            if transformation in legal_ops_by_type[col_types[col]]
        ] + [('TABLE',op) for op in whole_table_ops]
        random.shuffle(actions)
        sequence = () if sequence_to_extend is None else sequence_to_extend
        seen_transformation_classes_per_col = defaultdict(list)
        for (op,col) in sequence:
            if col == 'TABLE' or op.startswith('fill_'):
                continue
            trans_class = op.split("_")[0]
            seen_transformation_classes_per_col[col].append(trans_class)

        if sequence_to_extend is not None and df.isna().any(axis=None):
            fill_methods = get_fill_combinations(df, col_types, legal_ops_by_type)
            sequence = random.choice(fill_methods)

        for _ in range(sequence_length):
            found_action = False
            while not found_action and len(actions) > 0:
                col, transformation = actions.pop()
                if transformation == "isolationForest" and any(f_n == "isolationForest" for f_n, c in sequence):
                    continue
                if transformation == "drop_duplicates" and any(f_n == "drop_duplicates" for f_n, c in sequence):
                    continue

                trans_class = transformation.split("_")[0]
                if trans_class not in seen_transformation_classes_per_col[col]:
                    seen_transformation_classes_per_col[col].append(trans_class)
                    sequence = sequence + ((transformation, col),)
                    found_action = True
        curr_df = apply_data_preparations_seq(df, sequence, transformations_dict)
        new_ate = calculate_ate_linear_regression_lstsq(curr_df, 'treatment', 'outcome', common_causes)
        return sequence, new_ate


