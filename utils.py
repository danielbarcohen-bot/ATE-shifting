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
    df_ = df_.dropna()
    # df_.fillna(value=df_.mean(), inplace=True)  # filling the missing values
    # df_.fillna(value=df_.mode().loc[0], inplace=True)
    # model = CausalModel(
    #     data=df_,
    #     treatment='treatment',
    #     outcome='outcome',
    #     common_causes=common_causes
    # )
    #
    # return calculate_ate(model)
    return calculate_ate_linear_regression_lstsq(df_, 'treatment', 'outcome', common_causes)


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


def bin_5_equal_width(s):
    if s.nunique() <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index)
    bins = pd.cut(s, bins=5, labels=False, duplicates="drop")
    return bins.fillna(0)


def log1p_safe(s):
    return np.log1p(s.clip(lower=0))


def square(s):
    return s ** 2


def sqrt_clip(s):
    return np.sqrt(s.clip(lower=0))


def exp_norm(s):
    z = (s - s.mean()) / (s.std() + 1e-8)
    e = np.exp(z)
    return e / (e.mean() + 1e-8)


def get_transformations():
    strategies = {
        "fill_median": fill_median,
        "fill_min": fill_min,
        "zscore_clip_3": zscore_clip_3,
        "bin_2": bin_2,
        # "bin_5_equal_width": bin_5_equal_width,
        # "log1p_safe": log1p_safe,
        # "square": square,
        # "sqrt_clip": sqrt_clip,
        # "exp_norm": exp_norm
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
    transformations = get_transformations()
    for func_name, col in seq_arr:
        df_[col] = transformations[func_name](df_[col])
    return df_

def get_moves_and_moveBit(common_causes, transformations_names):
    bit_map = {}
    counter = 0
    for f in transformations_names:
        group = f.split('_')[0]
        for c in common_causes:
            if (group, c) not in bit_map:
                bit_map[(group, c)] = counter
                counter += 1

    # Pre-calculate moves: (func, col, bit_value)
    # bit_value is 2^counter (e.g., 1, 2, 4, 8, 16...)
    fast_moves = []
    for c in common_causes:
        for f in transformations_names:
            group = f.split('_')[0]
            bit_pos = bit_map[(group, c)]
            fast_moves.append((f, c, 1 << bit_pos))
    return fast_moves

# def calculate_ate_linear_regression_algebra(df: pd.DataFrame, treatment: str, outcome: str , common_causes: List[str]):
#     Y = df[outcome].values.reshape(-1, 1)
#     T = df[treatment].values.reshape(-1, 1)
#
#     # Confounders
#     X = df[common_causes].values
#
#     # Add intercept
#     X = np.hstack([np.ones((X.shape[0], 1)), X])
#
#     # Combine treatment and confounders
#     X_full = np.hstack([T, X])
#
#     # Use pseudo-inverse to handle singular/collinear columns
#     beta = np.linalg.pinv(X_full) @ Y
#
#     # First coefficient is treatment effect
#     ate = beta[0, 0]
#
#     return ate


def bin_sequences(data, num_bins=10):
    values = np.array([float(v) for seq, v in data])

    # Step 1 — compute histogram correctly
    counts, bin_edges = np.histogram(values, bins=num_bins)

    # Safety: make absolutely sure bin edges are sorted
    bin_edges = np.sort(bin_edges)

    # Step 2 — assign values to bins
    bin_indices = np.digitize(values, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Build result bins
    result = []
    for i in range(num_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        bucket = []
        for (seqs, v), idx in zip(data, bin_indices):
            if idx == i:
                bucket.append((seqs, float(v)))
        result.append({"range": (low, high), "items": bucket})

    return result


def find_interesting(entries, threshold=2 , round_after_n_digit=3):
    """
    entries: list of tuples like (lst, float_num)

    Returns all (lst, float_num) pairs where:
    - len(lst) > threshold
    - no entry exists with the same rounded float_num (3 decimals) and len(lst) <= threshold
    """

    from collections import defaultdict

    # Group entries by the float rounded to 3 decimals
    groups = defaultdict(list)
    for lst, num in entries:
        r = round(num, round_after_n_digit)
        groups[r].append((lst, num))

    interesting = []

    for rnum, items in groups.items():
        # Check if any small (len<=2) list exists with this rounded number
        has_small = any(len(lst) <= threshold for lst, num in items)

        # Keep only big lists if no small list exists
        if not has_small:
            for lst, num in items:
                if len(lst) > threshold:
                    interesting.append((lst, num))

    return interesting


import pandas as pd
import numpy as np
from typing import List


def calculate_ate_linear_regression_lstsq(df: pd.DataFrame, treatment: str, outcome: str, common_causes: List[str]):
    # Extract outcome variable
    Y = df[outcome].values

    # Extract treatment and confounders
    T = df[treatment].values.reshape(-1, 1)
    X_confounders = df[common_causes].values

    # 1. Create the full design matrix (X_full)
    # The columns must be in the order: [Treatment, Intercept, Confounder1, Confounder2, ...]

    # Add intercept column (a column of ones)
    X_intercept = np.ones((df.shape[0], 1))

    # Combine T, Intercept, and Confounders
    # This forms the X_full matrix for the regression: Y = beta0*T + beta1*Intercept + beta2*C1 + ...
    X_full = np.hstack([T, X_intercept, X_confounders])

    # NOTE: The intercept should be the *second* column if you want the treatment effect
    # to remain the *first* coefficient (beta[0, 0]).

    # 2. Use numpy.linalg.lstsq for the least-squares solution
    # beta will be the vector of coefficients: [ATE, Intercept_Coeff, Confounder1_Coeff, ...]
    # The [0] index extracts the coefficients array
    beta, residuals, rank, singular_values = np.linalg.lstsq(X_full, Y, rcond=None)

    # First coefficient is the Average Treatment Effect (ATE)
    ate = beta[0]

    return ate
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
#
#
# def analyze_ate_risk(df, treatment_col, outcome_col, mandatory_keep_features=[]):
#     """
#     Analyzes the worst-case ATE bias risk for every feature if it were dropped.
#
#     This function calculates the ROBUST STATISTICAL BOUND risk, which is
#     invariant to changes in other features (X_k) and accounts for sample noise.
#
#     Args:
#         df (DataFrame): The input data.
#         treatment_col (str): Name of the treatment column (T).
#         outcome_col (str): Name of the outcome column (Y).
#         mandatory_keep_features (list): Features that cannot be dropped
#                                         (e.g., known strong confounders).
#     """
#
#     # 1. Setup
#     features = [c for c in df.columns if c not in [treatment_col, outcome_col]]
#     n = len(df)
#
#     # Calculate Global Scaling Factor (Sigma_Y / Sigma_T)
#     sigma_y = df[outcome_col].std()
#     sigma_t = df[treatment_col].std()
#
#     if sigma_t == 0:
#         scale_factor = 0
#     else:
#         scale_factor = sigma_y / sigma_t
#
#     # Statistical Floor: Accounts for sample noise (2 * StD of correlation estimate)
#     statistical_floor = 2.0 / np.sqrt(n)
#
#     analysis_data = []
#
#     # 2. Analyze every feature
#     corr_matrix = df[[treatment_col, outcome_col] + features].corr()
#
#     for feat in features:
#         # Rho_ZT: Correlation between Feature and Treatment
#         rho_zt = corr_matrix.loc[feat, treatment_col]
#
#         # Rho_ZY: Correlation between Feature and Outcome
#         rho_zy = corr_matrix.loc[feat, outcome_col]
#
#         if np.isnan(rho_zt): rho_zt = 0.0
#         if np.isnan(rho_zy): rho_zy = 0.0
#
#         # THE FORMULA: Robust Statistical Worst-Case Bias Bound
#         # We use the maximal possible correlation (point estimate + floor)
#         risk_zt = abs(rho_zt) + statistical_floor
#         risk_zy = abs(rho_zy) + statistical_floor
#
#         worst_case_bias = risk_zy * risk_zt * scale_factor
#
#         is_kept = feat in mandatory_keep_features
#
#         analysis_data.append({
#             'feature': feat,
#             'corr_with_treatment': rho_zt,
#             'corr_with_outcome': rho_zy,
#             'worst_case_bias_risk': worst_case_bias,
#             'is_mandatory_kept': is_kept
#         })
#
#     detail_df = pd.DataFrame(analysis_data)
#
#     kept_df = detail_df[detail_df['is_mandatory_kept'] == True]
#
#     summary = {
#         'total_features': len(features),
#         'features_to_keep': kept_df['feature'].tolist(),
#         'statistical_floor_used': statistical_floor
#     }
#
#     # Sort all features by risk (lowest first) for budget analysis
#     return summary, detail_df.sort_values(by='worst_case_bias_risk', ascending=True)
#
#
# def find_droppable_features_by_budget(detail_df, max_bias_budget):
#     """
#     Determines the maximum number of features that can be dropped while staying
#     under a specified total ATE bias budget. This is the core function for the user's request.
#
#     Args:
#         detail_df (DataFrame): The detailed risk output from analyze_ate_risk.
#         max_bias_budget (float): The maximum allowed sum of bias risk (K).
#     Returns:
#         tuple: (list of features to drop, cumulative risk)
#     """
#
#     # 1. Filter out features that MUST be kept
#     drop_candidates = detail_df.copy()
#
#     if drop_candidates.empty:
#         return [], 0.0
#
#     # Candidates are already sorted by risk (lowest first) from analyze_ate_risk
#
#     dropped_features = []
#     cumulative_risk = 0.0
#
#     # 2. Iteratively add features until the budget is hit
#     for index, row in drop_candidates.iterrows():
#         feature_risk = row['worst_case_bias_risk']
#
#         # Check if adding this feature exceeds the budget
#         if cumulative_risk + feature_risk > max_bias_budget:
#             break  # Budget exceeded, stop dropping
#
#         # Add feature and update cumulative risk
#         dropped_features.append(row['feature'])
#         cumulative_risk += feature_risk
#
#     return dropped_features, cumulative_risk
