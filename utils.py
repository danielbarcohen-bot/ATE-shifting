import hashlib
from typing import List

import numpy as np
import pandas as pd
from dowhy import CausalModel


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


def fill_median(s):
    return s.fillna(s.median())


def fill_min(s):
    return s.fillna(s.min())


def zscore_clip_3(s):
    return s.where(np.abs((s - s.mean()) / (s.std() + 1e-8)) < 3, s.mean())


def bin_2(s):
    if s.nunique() == 1:
        return s
    return pd.qcut(s, q=2, labels=False, duplicates="drop")


def get_transformations():
    strategies = {
        "fill_median": fill_median,
        "fill_min": fill_min,
        "zscore_clip_3": zscore_clip_3,
        "bin_2": bin_2,
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


def df_signature_fast_rounds(df: pd.DataFrame, cols: List[str], decimals=10) -> str:
    """
    Return an order-insensitive, value-sensitive signature for a subset of columns,
    rounding floats to be robust to tiny process-level differences.
    """
    # Create a copy to avoid modifying the original df
    df_to_hash = df[cols].copy()

    # Identify float columns and round them to a fixed precision
    float_cols = df_to_hash.select_dtypes(include='float').columns
    df_to_hash[float_cols] = df_to_hash[float_cols].round(decimals=decimals)

    # Hash each row
    row_hashes = pd.util.hash_pandas_object(df_to_hash, index=False).values

    # Sort hashes so row order doesn’t matter
    row_hashes.sort()

    # Hash the sorted hashes
    return hashlib.sha1(row_hashes.tobytes()).hexdigest()

def apply_data_preparations_seq(df: pd.DataFrame, seq_arr):
    df_ = df.copy()
    for func_name, col in seq_arr:
        df_[col] = get_transformations()[func_name](df_[col])
    return df_
