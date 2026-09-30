import json
from typing import List, Dict

import numpy as np
import pandas as pd

from data_loader import TwinsDataLoader, LalondeDataLoader, IHDPDataLoader

twins_data_set = TwinsDataLoader().load_data()
lalonde_data_set = LalondeDataLoader().load_data()
ihdp_data_set = IHDPDataLoader().load_data()

SYSTEM_PROMPT_CLAUDE = """<task_description>
You are a causal inference optimization assistant. Your goal is to steer the Average Treatment Effect (ATE) of a dataset toward a target value by selecting optimal data transformations.

<input_specifications>
You will receive:
1. Statistical summary of the dataset including column data types (Numerical, Ordinal, Categorical, Binary)
2. Current ATE value (calculated via linear regression)
3. Target ATE value
4. Epsilon (acceptable range: target ± epsilon)
5. Sample rows for reference
</input_specifications>

<allowed_operations_by_column_type>
You MUST respect column data types when selecting operations. Never apply an operation to a column type not explicitly listed below.

1. Numerical Columns:
   - Imputation: fill_mean, fill_median
   - Binning: bin_equal_frequency_2, bin_equal_frequency_5, bin_equal_frequency_10, bin_equal_width_2, bin_equal_width_5, bin_equal_width_10
   - Normalization & Scaling: norm_min_max, norm_log, zscore_clip_3, winsorize
   - Outlier Filtering: zscore_filter_3, IQR

2. Ordinal Columns:
   - Imputation: fill_mode, fill_median
   - Binning: bin_equal_frequency_2, bin_equal_frequency_5, bin_equal_frequency_10, bin_equal_width_2, bin_equal_width_5, bin_equal_width_10
   - Normalization & Scaling: norm_min_max, norm_log, zscore_clip_3, winsorize
   - Outlier Filtering: zscore_filter_3, IQR

3. Categorical Columns:
   - Imputation: fill_mode ONLY
   - No binning, scaling, or filtering operations allowed.

4. Binary Columns:
   - Imputation: fill_mode ONLY
   - No binning, scaling, or filtering operations allowed.

5. Dataset-Wide Operations (Apply to the whole table):
   - isolationForest: Anomaly/outlier removal across the whole dataset
   - drop_duplicates: Remove duplicate rows
   - fill_drop_na: Drop any rows containing missing values
</allowed_operations_by_column_type>

<constraints>
CRITICAL RULES:
1. Whole-Dataset Operations: For 'isolationForest', 'drop_duplicates', and 'fill_drop_na', set the "column" field to "TABLE".
2. Column Type Validity: Strictly check each column's data type before applying any operation.
3. Per-Column Limits:
   - At most ONE Imputation operation per column.
   - At most ONE Binning operation per column.
   - At most ONE Normalization/Scaling operation per column.
   - At most ONE Outlier Filtering operation per column.
4. Protected Columns: NEVER apply transformations to the treatment or outcome variables.
</constraints>

<output_format>
Return ONLY a valid JSON array of transformation steps in execution order. Do not include markdown code fences or conversational text.

Example format:
[
  {"column": "TABLE", "operation": "drop_duplicates"},
  {"column": "age", "operation": "fill_median"},
  {"column": "age", "operation": "zscore_clip_3"},
  {"column": "category_code", "operation": "fill_mode"},
  {"column": "income", "operation": "bin_equal_frequency_5"},
  {"column": "TABLE", "operation": "isolationForest"}
]

If the target ATE is already within target ± epsilon, return an empty array: []
</output_format>
</task_description>"""

DO_NOT_THINK = "\nDo not include any text before or after the JSON array."


def format_dataset_context(
        df: pd.DataFrame,
        current_ate: float,
        target_ate: float,
        epsilon: float,
        treatment_col: str,
        outcome_col: str,
) -> str:
    """Single source of truth for dataset overview and feature statistics.

    Ensures both the main prompt and few-shot examples use the exact same layout.
    """
    col_types = getattr(df, "attrs", {}).get("col_types", {})
    covariate_cols = [
        c for c in df.columns if c not in [treatment_col, outcome_col]
    ]

    n_rows = len(df)
    n_duplicates = int(df.duplicated().sum())
    rows_with_na = int(df.isna().any(axis=1).sum())
    na_pct = (rows_with_na / n_rows) if n_rows > 0 else 0.0

    diff = abs(target_ate - current_ate)
    direction = "INCREASE" if target_ate > current_ate else "DECREASE"

    lines = [
        "Dataset Overview:",
        f"- Size: {n_rows:,} rows × {len(covariate_cols)} covariates",
        f"- Treatment: {treatment_col} | Outcome: {outcome_col}",
        f"- Duplicate Rows: {n_duplicates:,}",
        f"- Rows with Missing Values: {rows_with_na:,} ({na_pct:.1%})",
        "",
        f"Current ATE: {current_ate:.6f}",
        f"Target ATE: {target_ate:.6f}",
        f"Epsilon: {epsilon:.6f}",
        f"Direction: Need to {direction} ATE by {diff:.6f}",
        "",
        "<feature_statistics>",
    ]

    for col in covariate_cols:
        c_type = col_types.get(col, "Numerical")
        series = df[col]

        n_missing = int(series.isna().sum())
        missing_pct = (n_missing / n_rows) if n_rows > 0 else 0.0

        lines.append(f"Column: '{col}' | Type: {c_type}")
        lines.append(f"  Missing Values: {n_missing:,} ({missing_pct:.1%})")

        if c_type in ["Numerical", "Ordinal"]:
            try:
                s_min, s_max = series.min(), series.max()
                s_mean, s_std = series.mean(), series.std()
                s_skew = series.skew()
                p5, p95 = series.quantile(0.05), series.quantile(0.95)

                corr_out = (
                    df[[col, outcome_col]].corr().iloc[0, 1]
                    if pd.api.types.is_numeric_dtype(series)
                       and pd.api.types.is_numeric_dtype(df[outcome_col])
                    else np.nan
                )
                corr_trt = (
                    df[[col, treatment_col]].corr().iloc[0, 1]
                    if pd.api.types.is_numeric_dtype(series)
                       and pd.api.types.is_numeric_dtype(df[treatment_col])
                    else np.nan
                )

                corr_out_str = (
                    f"{corr_out:+.3f}" if not pd.isna(corr_out) else "N/A"
                )
                corr_trt_str = (
                    f"{corr_trt:+.3f}" if not pd.isna(corr_trt) else "N/A"
                )

                lines.append(
                    f"  Range: [{s_min:.3f}, {s_max:.3f}] | Mean: {s_mean:.3f} ±"
                    f" {s_std:.3f}"
                )
                lines.append(
                    f"  Skew: {s_skew:.2f} | P5-P95: [{p5:.3f}, {p95:.3f}]"
                )
                lines.append(
                    f"  Corr(outcome): {corr_out_str} | Corr(treatment):"
                    f" {corr_trt_str}"
                )
            except Exception:
                lines.append(
                    "  [Summary calculation skipped due to uncastable values]"
                )

        elif c_type in ["Categorical", "Binary"]:
            n_unique = series.nunique(dropna=True)
            mode_series = series.mode()
            mode_val = mode_series.iloc[0] if not mode_series.empty else "N/A"
            lines.append(
                f"  Unique Values: {n_unique:,} | Mode: '{mode_val}'"
            )

    lines.append("</feature_statistics>")
    return "\n".join(lines)


def create_few_shot_example(
        df: pd.DataFrame,
        current_ate: float,
        target_ate: float,
        epsilon: float,
        treatment_col: str,
        outcome_col: str,
        transformations: List[Dict[str, str]],
        explanation: str = "",
) -> str:
    """Generates a few-shot example matching the dataset context layout."""
    context = format_dataset_context(
        df, current_ate, target_ate, epsilon, treatment_col, outcome_col
    )

    parts = [context]
    if explanation:
        parts.append(f"<reasoning>\n{explanation}\n</reasoning>")
    parts.append(
        f"<solution>\n{json.dumps(transformations, indent=2)}\n</solution>\n"
    )

    return "\n".join(parts)


def create_compact_steering_prompt(
        df: pd.DataFrame,
        current_ate: float,
        target_ate: float,
        epsilon: float,
        treatment_col: str,
        outcome_col: str,
        few_shots_prompt: str = "",
) -> str:
    """Generates the main user prompt matching the exact layout of few-shot examples."""
    context = format_dataset_context(
        df, current_ate, target_ate, epsilon, treatment_col, outcome_col
    )

    header = "Your goal is to steer the Average Treatment Effect (ATE) of a dataset toward a target value by selecting optimal data transformations."

    few_shots_block = f"\n{few_shots_prompt}\n" if few_shots_prompt else "\n"

    instructions = (
        "Propose transformations to steer ATE to target while strictly"
        " respecting column types and rules.\nReturn ONLY a JSON array in"
        ' execution order:\n[{"column": "col_name_or_TABLE", "operation":'
        ' "op_name"}, ...]'
    )

    return f"{header}\n{few_shots_block}{context}\n\n{instructions}"


def create_few_shots_prompt(few_shots: List[str]) -> str:
    """
    Add few-shot examples to the compact prompt.
    """

    examples_section = "<examples>\n"
    for example in few_shots:
        examples_section += "<example>\n" + example + "</example>\n"
    examples_section += "</examples>\n\n"
    examples_section += "=" * 70 + "\n"

    return examples_section


# FEW_SHOT_EXAMPLE_TWINS = create_few_shot_example(twins_data_set, 0.06, 0.0019, 0.000001, 'treatment', 'outcome', [
#     {'column': 'adequacy', 'operation': 'zscore_clip_3'},
#     {'column': 'lung', 'operation': 'bin_equal_frequency_2'},
#     {'column': 'wt', 'operation': 'bin_equal_frequency_2'}
# ])
FEW_SHOT_EXAMPLE_IHDP = create_few_shot_example(ihdp_data_set, 3.92, 4.2, 0.005, 'treatment', 'outcome', [
    {'column': 'x1', 'operation': 'bin_equal_frequency_2'},
    {'column': 'x3', 'operation': 'bin_equal_frequency_2'},
    {'column': 'x5', 'operation': 'bin_equal_width_2'},
    {'column': 'TABLE', 'operation': 'isolationForest'}
])
FEW_SHOT_EXAMPLE_LALONDE = create_few_shot_example(lalonde_data_set, 1671, 2000, 1, 'treatment', 'outcome', [
    {'column': 'education', 'operation': 'bin_equal_width_10'},
    {'column': 'TABLE', 'operation': 'isolationForest'},
    {'column': 'age', 'operation': 'bin_equal_frequency_2'},
    {'column': 'education', 'operation': 'IQR'}
])

# if __name__ == '__main__':
#     print(create_compact_steering_prompt(lalonde_data_set, 0.06, -0.06, 0.06,'treatment', 'outcome'))
#     few_shot_prompt = create_compact_steering_prompt(twins_data_set, 0.06, -0.06, 0.06,'treatment', 'outcome',create_few_shots_prompt([FEW_SHOT_EXAMPLE_IHDP, FEW_SHOT_EXAMPLE_LALONDE]))
#     print(few_shot_prompt)
