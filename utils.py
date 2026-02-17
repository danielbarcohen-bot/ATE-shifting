import hashlib
from typing import List

import numpy as np
import pandas as pd
from dowhy import CausalModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LassoCV, LogisticRegression
from sklearn.model_selection import KFold


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


# fill
def fill_median(s):
    return s.fillna(s.median())


def fill_min(s):
    return s.fillna(s.min())


# bin
def bin_equal_frequency_2(s):
    if s.nunique() < 2:
        return s
    return pd.qcut(s, q=2, labels=False, duplicates="drop")


def bin_equal_frequency_5(s):
    if s.nunique() < 5:
        return s
    return pd.qcut(s, q=2, labels=False, duplicates="drop")


def bin_equal_frequency_10(s):
    if s.nunique() < 10:
        return s
    return pd.qcut(s, q=2, labels=False, duplicates="drop")


def bin_equal_width_2(s):
    if s.nunique() < 2:
        return s  # pd.Series(np.zeros(len(s)), index=s.index)
    bins = pd.cut(s, bins=2, labels=False, include_lowest=True)
    return bins


def bin_equal_width_5(s):
    if s.nunique() < 5:
        return s  # pd.Series(np.zeros(len(s)), index=s.index)
    bins = pd.cut(s, bins=5, labels=False, include_lowest=True)
    return bins


def bin_equal_width_10(s):
    if s.nunique() < 10:
        return s  # pd.Series(np.zeros(len(s)), index=s.index)
    bins = pd.cut(s, bins=10, labels=False, include_lowest=True)
    return bins


# normalizing

def min_max_norm(s: pd.Series) -> pd.Series:
    min_v = s.min()
    max_v = s.max()

    if min_v == max_v:
        return pd.Series(0.0, index=s.index)

    return (s - min_v) / (max_v - min_v)


def log_norm(s: pd.Series) -> pd.Series:
    return np.sign(s) * np.log1p(np.abs(s))


# outlier detection
def zscore_clip_3(s):
    return s.where(np.abs((s - s.mean()) / (s.std() + 1e-8)) < 3, s.mean())


def winsorize(s: pd.Series, lower_quantile=0.01, upper_quantile=0.99) -> pd.Series:
    lower = s.quantile(lower_quantile)
    upper = s.quantile(upper_quantile)
    return s.clip(lower, upper)


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


def apply_data_preparations_seq(df: pd.DataFrame, seq_arr, transformations_dict):
    df_ = df.copy()
    for func_name, col in seq_arr:
        df_[col] = transformations_dict[func_name](df_[col])
    return df_


def list_seq_to_tuple_seq(list_seq):
    tuple_seq = ()
    for seq_dict in list_seq:
        tuple_seq = tuple_seq + ((seq_dict["operation"], seq_dict["column"]),)
    return tuple_seq


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


def find_interesting(entries, threshold=2, round_after_n_digit=3):
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


def manual_dml_ate(df, outcome_col='outcome', treatment_col='treatment'):
    X = df.drop(columns=[outcome_col, treatment_col])
    y = df[outcome_col].values
    T = df[treatment_col].values

    y_res = np.zeros_like(y, dtype=float)
    T_res = np.zeros_like(T, dtype=float)

    # Use 2-fold cross-fitting for maximum speed
    # random_state=42 makes it deterministic
    kf = KFold(n_splits=2, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        # LassoCV is extremely fast compared to Random Forest
        model_y = LassoCV(cv=3).fit(X.iloc[train_idx], y[train_idx])
        model_t = LassoCV(cv=3).fit(X.iloc[train_idx], T[train_idx])

        y_res[test_idx] = y[test_idx] - model_y.predict(X.iloc[test_idx])
        T_res[test_idx] = T[test_idx] - model_t.predict(X.iloc[test_idx])

    # Final step: Simple Linear Regression on residuals
    final_model = LinearRegression(fit_intercept=False).fit(T_res.reshape(-1, 1), y_res)
    return final_model.coef_[0]


def manual_dr_ate(df, outcome_col='outcome', treatment_col='treatment'):
    X = df.drop(columns=[outcome_col, treatment_col])
    y = df[outcome_col].values
    T = df[treatment_col].values
    n = len(y)

    # 1. Propensity Score (Probability of Treatment) -> Fast & Deterministic
    clf = LogisticRegression(max_iter=1000).fit(X, T)
    e = np.clip(clf.predict_proba(X)[:, 1], 0.01, 0.99)  # Clip to avoid division by zero

    # 2. Outcome Models -> Fast & Deterministic
    model_0 = LassoCV(cv=3).fit(X[T == 0], y[T == 0])
    model_1 = LassoCV(cv=3).fit(X[T == 1], y[T == 1])

    mu_0 = model_0.predict(X)
    mu_1 = model_1.predict(X)

    # 3. Individual AIPW Scores (The "Double Robust" Magic)
    # We calculate the treatment effect for every single row
    scores = (mu_1 + (T * (y - mu_1) / e)) - (mu_0 + ((1 - T) * (y - mu_0) / (1 - e)))

    # 4. Average Treatment Effect (ATE)
    ate = np.mean(scores)

    return ate