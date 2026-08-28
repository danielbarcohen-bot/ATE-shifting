import hashlib
from typing import List

import numpy as np
import pandas as pd
from dowhy import CausalModel
from econml.dml import LinearDML
from econml.dr import DRLearner
from scipy.stats._mstats_basic import winsorize
from sklearn.ensemble import IsolationForest

from linear_ate_calculator import LinearATEModel


def calculate_ate(model: CausalModel):
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    estimate = model.estimate_effect(identified_estimand,
                                     method_name="backdoor.linear_regression", test_significance=True)

    return estimate.value


def get_baseline_ate(common_causes, df):
    df_ = df.copy()
    df_ = df_.dropna()
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
    s = df[col]
    if s.nunique() <= k:
        return df

    try:
        binnded_col = pd.qcut(s, q=k, labels=False)
    except:
        return df

    df[col] = binnded_col
    return df


def bin_equal_frequency_2(df, col) -> pd.DataFrame:
    return bin_equal_frequency_k(df, col, 2)


def bin_equal_frequency_5(df, col) -> pd.DataFrame:
    return bin_equal_frequency_k(df, col, 5)


def bin_equal_frequency_10(df, col) -> pd.DataFrame:
    return bin_equal_frequency_k(df, col, 10)


def bin_equal_width_k(df, col, k) -> pd.DataFrame:
    s = df[col]
    if s.nunique() <= k:
        return df
    df[col] = pd.cut(s, bins=k, labels=False, include_lowest=True)
    return df


def bin_equal_width_2(df, col) -> pd.DataFrame:
    return bin_equal_width_k(df, col, 2)


def bin_equal_width_5(df, col) -> pd.DataFrame:
    return bin_equal_width_k(df, col, 5)


def bin_equal_width_10(df, col) -> pd.DataFrame:
    return bin_equal_width_k(df, col, 10)


# normalizing
def min_max_norm(df, col) -> pd.DataFrame:
    s = df[col]

    min_v = s.min()
    max_v = s.max()

    if min_v == max_v:
        df[col] = pd.Series(0.0, index=s.index)
        return df

    df[col] = (s - min_v) / (max_v - min_v)
    return df


def log_norm(df, col) -> pd.DataFrame:
    s = df[col]

    df[col] = np.sign(s) * np.log1p(np.abs(s))
    return df


# outlier detection
def zscore_clip_3(df, col) -> pd.DataFrame:
    s = df[col]

    df[col] = s.where(np.abs((s - s.mean()) / (s.std() + 1e-8)) < 3, s.mean())
    return df


def zscore_filter_3(df, col) -> pd.DataFrame:
    s = df[col]

    if s.nunique() <= 2:
        return df

    z_score = np.abs((s - s.mean()) / (s.std() + 1e-8))
    mask = (z_score < 3) & (~s.isna())

    return df[mask]


def winsorize_aux(df, col) -> pd.DataFrame:
    if df.empty:
        return df
    w = winsorize(df[col].to_numpy(), limits=[0.01, 0.01])
    df[col] = pd.Series(
        np.asarray(w),
        index=df.index,
        name=col
    )

    return df


def IQR(df, col) -> pd.DataFrame:
    s = df[col]
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return df
    df = df[(s >= q1 - 1.5 * iqr) & (s <= q3 + 1.5 * iqr)]
    return df


def isolationForest(df, col) -> pd.DataFrame:
    if df.empty:
        return df
    df_ = df.copy()
    common_causes = df.columns.difference(["treatment", "outcome", "replica_id"], sort=False).tolist()
    iso = IsolationForest(random_state=42)
    df_['outlier_label'] = iso.fit_predict(df_[common_causes])
    df_clean = df_[df_['outlier_label'] == 1].drop(columns=['outlier_label'])

    if df_clean.empty:
        return df

    return df_clean


def dropDuplicates(df, col) -> pd.DataFrame:
    return df.drop_duplicates()


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


def prepare_inference_matrix(df: pd.DataFrame, common_causes: List[str]) -> pd.DataFrame:
    categorical_causes = df.attrs.get('categorical_causes', [])
    if len(categorical_causes) == 0:
        return df[common_causes]

    X_encoded = pd.get_dummies(df[common_causes], columns=categorical_causes, drop_first=True, dtype=int)
    return X_encoded


def calculate_ate_linear_regression_lstsq(df: pd.DataFrame, treatment: str, outcome: str, common_causes: List[str]):
    x = prepare_inference_matrix(df, common_causes)
    return LinearATEModel(x, df[treatment], df[outcome]).ate


def calculate_ate_with_uncertainty(df: pd.DataFrame, treatment: str, outcome: str, common_causes: List[str]):
    x = prepare_inference_matrix(df, common_causes)
    model = LinearATEModel(x, df[treatment], df[outcome])
    ate = model.ate
    ci = model.ci()
    return {'ate': ate, 'ci': ci}


def calculate_ate_dml(df, outcome_col='outcome', treatment_col='treatment'):
    common_causes = df.columns.difference(["treatment", "outcome"], sort=False)
    y = df[outcome_col].values
    T = df[treatment_col].values
    W = prepare_inference_matrix(df, common_causes).values#df.drop(columns=[outcome_col, treatment_col]).values


    is_outcome_binary = len(np.unique(y)) == 2
    dml_model = LinearDML(discrete_outcome=is_outcome_binary, discrete_treatment=True, random_state=42)

    dml_model.fit(y, T, W=W)

    ate = dml_model.ate()
    return ate


def calculate_ate_dr(df, outcome_col='outcome', treatment_col='treatment'):
    common_causes = df.columns.difference(["treatment", "outcome"], sort=False)

    y = df[outcome_col].values
    T = df[treatment_col].values
    X = prepare_inference_matrix(df, common_causes).values#df.drop(columns=[outcome_col, treatment_col]).values

    is_outcome_binary = len(np.unique(y)) == 2

    dr_model = DRLearner(discrete_outcome=is_outcome_binary, random_state=42)
    dr_model.fit(y, T, W=X)
    return dr_model.ate()
