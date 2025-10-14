from typing import List

from dowhy import CausalModel
import numpy as np
import pandas as pd
import hashlib


def calculate_ate(model: CausalModel):
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    estimate = model.estimate_effect(identified_estimand,
                                     method_name="backdoor.linear_regression", test_significance=True)

    return estimate.value


def get_base_line(common_causes, df):
    df_ = df.copy()
    df_.fillna(value=df_.mean(), inplace=True)  # filling the missing values
    df_.fillna(value=df_.mode().loc[0], inplace=True)
    model = CausalModel(
        data=df_,
        treatment='treatment',
        outcome='outcome',
        common_causes=common_causes
    )

    return calculate_ate(model)


def get_transformations():
    strategies = {
        "fill_median": lambda s: s.fillna(s.median()),
        "fill_min": lambda s: s.fillna(s.min()),
        "zscore_clip_3": lambda s: s.where(np.abs((s - s.mean()) / (s.std() + 1e-8)) < 3, s.mean()),
        "bin_2": lambda s: s if s.nunique() == 1 else pd.qcut(s, q=2, labels=False, duplicates="drop"),

    }

    return strategies


def df_signature(df: pd.DataFrame):
    return frozenset([tuple(row) for row in df.to_numpy()])


def df_signature_fast(df: pd.DataFrame, cols: List[str]) -> str:
    """Return an order-insensitive, value-sensitive signature for a subset of columns."""
    # Hash each row (subset of columns)
    row_hashes = pd.util.hash_pandas_object(df[cols], index=False).values
    # Sort hashes so row order doesn’t matter
    row_hashes.sort()
    # Hash the sorted hashes — identical if same rows (with duplicates)
    return hashlib.sha1(row_hashes.tobytes()).hexdigest()


def apply_data_preparations_seq(df: pd.DataFrame, seq_arr):
    df_ = df.copy()
    for func_name, col in seq_arr:
        df_[col] = get_transformations()[func_name](df_[col])
    return df_
