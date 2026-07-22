import hashlib
from typing import List

import numpy as np
import pandas as pd
from dowhy import CausalModel
from scipy import stats
from scipy.stats._mstats_basic import winsorize
from sklearn.ensemble import IsolationForest
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
def fill_median(df, col):
    df_ = df.copy()
    df_[col] = df_[col].fillna(df_[col].median())
    return df_


def fill_min(df, col):
    df_ = df.copy()
    df_[col] = df_[col].fillna(df_[col].min())
    return df_


# bin
def bin_equal_frequency_k(df, col, k) -> pd.DataFrame:
    df_ = df.copy()
    s = df_[col]
    if s.nunique() <= k:
        return df_

    try:
        binnded_col = pd.qcut(s, q=k, labels=False)
    except:
        return df_

    df_[col] = binnded_col
    return df_


def bin_equal_frequency_2(df, col) -> pd.DataFrame:
    return bin_equal_frequency_k(df, col, 2)


def bin_equal_frequency_5(df, col) -> pd.DataFrame:
    return bin_equal_frequency_k(df, col, 5)


def bin_equal_frequency_10(df, col) -> pd.DataFrame:
    return bin_equal_frequency_k(df, col, 10)


def bin_equal_width_k(df, col, k) -> pd.DataFrame:
    df_ = df.copy()
    s = df_[col]
    if s.nunique() <= k:
        return df_
    df_[col] = pd.cut(s, bins=k, labels=False, include_lowest=True)
    return df_


def bin_equal_width_2(df, col) -> pd.DataFrame:
    return bin_equal_width_k(df, col, 2)


def bin_equal_width_5(df, col) -> pd.DataFrame:
    return bin_equal_width_k(df, col, 5)


def bin_equal_width_10(df, col) -> pd.DataFrame:
    return bin_equal_width_k(df, col, 10)


# normalizing
def min_max_norm(df, col) -> pd.DataFrame:
    df_ = df.copy()
    s = df_[col]

    min_v = s.min()
    max_v = s.max()

    if min_v == max_v:
        df_[col] = pd.Series(0.0, index=s.index)
        return df_

    df_[col] = (s - min_v) / (max_v - min_v)
    return df_


def log_norm(df, col) -> pd.DataFrame:
    df_ = df.copy()
    s = df_[col]

    df_[col] = np.sign(s) * np.log1p(np.abs(s))
    return df_


# outlier detection
def zscore_clip_3(df, col) -> pd.DataFrame:
    df_ = df.copy()
    s = df_[col]

    df_[col] = s.where(np.abs((s - s.mean()) / (s.std() + 1e-8)) < 3, s.mean())
    return df_


def zscore_filter_3(df, col) -> pd.DataFrame:
    df_ = df.copy()
    s = df_[col]

    if s.nunique() <= 2:
        return df_

    z_score = np.abs((s - s.mean()) / (s.std() + 1e-8))
    mask = (z_score < 3) & (~s.isna())

    return df_[mask]


def winsorize_aux(df, col) -> pd.DataFrame:
    df_ = df.copy()

    # lower = s.quantile(lower_quantile)
    # upper = s.quantile(upper_quantile)
    # df_[col] = s.clip(lower, upper)
    df_[col] = winsorize(df_[col], limits=[0.01, 0.01])
    return df_


def IQR(df, col) -> pd.DataFrame:
    df_ = df.copy()
    s = df_[col]

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return df
    df_ = df_[(s >= q1 - 1.5 * iqr) & (s <= q3 + 1.5 * iqr)]
    return df_


def isolationForest(df, col) -> pd.DataFrame:
    if df.empty:
        return df
    df_ = df.copy()
    common_causes = df.columns.difference(["treatment", "outcome"]).tolist()
    iso = IsolationForest(random_state=42)
    df_['outlier_label'] = iso.fit_predict(df_[common_causes])
    df_clean = df_[df_['outlier_label'] == 1].drop(columns=['outlier_label'])

    if df_clean.empty:
        return df

    return df_clean


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
        df_ = transformations_dict[func_name](df_, col)
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


def analyze_ate_search_space(seq_ates):
    """
    Groups ATE results into bucket ranges and extracts the shortest
    sequence for each range.
    """
    # 1. Setup DataFrame
    df = pd.DataFrame(seq_ates, columns=['sequence', 'ate'])
    df['length'] = df['sequence'].apply(len)

    # Remove inf/nan to prevent binning errors
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['ate'])

    if df.empty:
        return "No valid data to analyze."

    # 2. Auto-Calculate Bin Width (Freedman-Diaconis)
    ates = df['ate'].values
    q75, q25 = np.percentile(ates, [75, 25])
    iqr = q75 - q25

    # Calculate width; default to 0.1 if data is too tight
    bin_width = (2 * iqr * (len(ates) ** (-1 / 3))) if iqr > 1e-6 else 0.1

    # 3. Create the actual ranges (Buckets)
    # We create bins from the floor of the min to the ceil of the max
    min_ate, max_ate = df['ate'].min(), df['ate'].max()

    # Create an array of edge points for the bins
    bins = np.arange(min_ate - bin_width, max_ate + bin_width, bin_width)

    # Categorize the data into these ranges
    df['bucket_range'] = pd.cut(df['ate'], bins=bins)

    # 4. Extract the best (shortest) representative for every bucket
    summary = []
    for bucket, group in df.groupby('bucket_range', observed=True):
        # Find the shortest path in this specific range
        min_len = group['length'].min()
        shortest_path_row = group[group['length'] == min_len].iloc[0]

        summary.append({
            'bucket_range': bucket,
            'min_ate_in_bin': group['ate'].min(),
            'max_ate_in_bin': group['ate'].max(),
            'min_length': min_len,
            'best_sequence': shortest_path_row['sequence'],
            'count': len(group)
        })

    return pd.DataFrame(summary)


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

import statsmodels.api as sm
def calculate_ate_with_uncertainty(df: pd.DataFrame, treatment: str, outcome: str, common_causes: List[str]):
    X = sm.add_constant(df[[treatment] + common_causes])
    Y = df[outcome]

    # cov_type='HC1' gives you causal-inference-ready robust standard errors
    model = sm.OLS(Y, X).fit()#cov_type='HC1')

    ate = model.params[treatment]
    ate_se = model.bse[treatment]
    ci = model.conf_int().loc[treatment]
    p_val = model.pvalues[treatment]

    return {
        'ate': ate,
        'se': ate_se,
        'ci': (ci[0], ci[1]),
        'significant': p_val < 0.05
    }
    # Y = df[outcome].values
    # T = df[treatment].values.reshape(-1, 1)
    # X_confounders = df[common_causes].values
    # X_intercept = np.ones((df.shape[0], 1))
    # X_full = np.hstack([T, X_intercept, X_confounders])
    # X_full = np.asarray(X_full)
    # n, k = X_full.shape
    # beta, residuals, rank, singular_values = np.linalg.lstsq(X_full, Y, rcond=None)
    # ate = beta[0]
    # ssr = np.sum((Y - X_full @ beta) ** 2)
    # sigma_sq = ssr / (n - k)
    #
    # # 4. Calculate Variance-Covariance Matrix: sigma^2 * (X^T X)^-1
    # # This tells us how much each coefficient "wiggles"
    # xtx_inv = np.linalg.pinv(X_full.T @ X_full)
    # var_cov_matrix = sigma_sq * xtx_inv
    #
    # # 5. Extract Standard Error for ATE (the first diagonal element)
    # ate_se = np.sqrt(var_cov_matrix[0, 0])
    #
    # # 6. Calculate 95% Confidence Interval
    # # For 95%, we use a t-distribution critical value (approx 1.96)
    # t_critical = stats.t.ppf(0.975, df=n - k)
    # ci_lower = ate - (t_critical * ate_se)
    # ci_upper = ate + (t_critical * ate_se)
    #
    # return {
    #     'ate': ate,
    #     'se': ate_se,
    #     'ci': (ci_lower, ci_upper),
    #     'significant': not (ci_lower <= 0 <= ci_upper)
    # }


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
