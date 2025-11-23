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


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def analyze_ate_risk(df, treatment_col, outcome_col, mandatory_keep_features=[]):
    """
    Analyzes the worst-case ATE bias risk for every feature if it were dropped.

    This function calculates the ROBUST STATISTICAL BOUND risk, which is
    invariant to changes in other features (X_k) and accounts for sample noise.

    Args:
        df (DataFrame): The input data.
        treatment_col (str): Name of the treatment column (T).
        outcome_col (str): Name of the outcome column (Y).
        mandatory_keep_features (list): Features that cannot be dropped
                                        (e.g., known strong confounders).
    """

    # 1. Setup
    features = [c for c in df.columns if c not in [treatment_col, outcome_col]]
    n = len(df)

    # Calculate Global Scaling Factor (Sigma_Y / Sigma_T)
    sigma_y = df[outcome_col].std()
    sigma_t = df[treatment_col].std()

    if sigma_t == 0:
        scale_factor = 0
    else:
        scale_factor = sigma_y / sigma_t

    # Statistical Floor: Accounts for sample noise (2 * StD of correlation estimate)
    statistical_floor = 2.0 / np.sqrt(n)

    analysis_data = []

    # 2. Analyze every feature
    corr_matrix = df[[treatment_col, outcome_col] + features].corr()

    for feat in features:
        # Rho_ZT: Correlation between Feature and Treatment
        rho_zt = corr_matrix.loc[feat, treatment_col]

        # Rho_ZY: Correlation between Feature and Outcome
        rho_zy = corr_matrix.loc[feat, outcome_col]

        if np.isnan(rho_zt): rho_zt = 0.0
        if np.isnan(rho_zy): rho_zy = 0.0

        # THE FORMULA: Robust Statistical Worst-Case Bias Bound
        # We use the maximal possible correlation (point estimate + floor)
        risk_zt = abs(rho_zt) + statistical_floor
        risk_zy = abs(rho_zy) + statistical_floor

        worst_case_bias = risk_zy * risk_zt * scale_factor

        is_kept = feat in mandatory_keep_features

        analysis_data.append({
            'feature': feat,
            'corr_with_treatment': rho_zt,
            'corr_with_outcome': rho_zy,
            'worst_case_bias_risk': worst_case_bias,
            'is_mandatory_kept': is_kept
        })

    detail_df = pd.DataFrame(analysis_data)

    kept_df = detail_df[detail_df['is_mandatory_kept'] == True]

    summary = {
        'total_features': len(features),
        'features_to_keep': kept_df['feature'].tolist(),
        'statistical_floor_used': statistical_floor
    }

    # Sort all features by risk (lowest first) for budget analysis
    return summary, detail_df.sort_values(by='worst_case_bias_risk', ascending=True)


def find_droppable_features_by_budget(detail_df, max_bias_budget):
    """
    Determines the maximum number of features that can be dropped while staying
    under a specified total ATE bias budget. This is the core function for the user's request.

    Args:
        detail_df (DataFrame): The detailed risk output from analyze_ate_risk.
        max_bias_budget (float): The maximum allowed sum of bias risk (K).
    Returns:
        tuple: (list of features to drop, cumulative risk)
    """

    # 1. Filter out features that MUST be kept
    drop_candidates = detail_df.copy()

    if drop_candidates.empty:
        return [], 0.0

    # Candidates are already sorted by risk (lowest first) from analyze_ate_risk

    dropped_features = []
    cumulative_risk = 0.0

    # 2. Iteratively add features until the budget is hit
    for index, row in drop_candidates.iterrows():
        feature_risk = row['worst_case_bias_risk']

        # Check if adding this feature exceeds the budget
        if cumulative_risk + feature_risk > max_bias_budget:
            break  # Budget exceeded, stop dropping

        # Add feature and update cumulative risk
        dropped_features.append(row['feature'])
        cumulative_risk += feature_risk

    return dropped_features, cumulative_risk
