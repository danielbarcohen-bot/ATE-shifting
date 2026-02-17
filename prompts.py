import json
from typing import List, Dict

import pandas as pd

from data_loader import TwinsDataLoader, LalondeDataLoader

twins_data_set = TwinsDataLoader().load_data().dropna()
lalonde_data_set = LalondeDataLoader().load_data().dropna()

SYSTEM_PROMPT_CLAUDE = """<task_description>
You are a causal inference optimization assistant. Your goal is to steer the Average Treatment Effect (ATE) of a dataset toward a target value by selecting optimal data transformations.

<input_specifications>
You will receive:
1. Statistical summary of the dataset (NOT the full dataframe - it may be huge)
2. Current ATE value (calculated via linear regression)
3. Target ATE value
4. Epsilon (acceptable range: target ± epsilon)
5. Sample rows for reference
</input_specifications>

<available_transformations>
You can apply these transformations to columns:

Binning operations (mutually exclusive per column):
- bin_equal_frequency_2: Split into 2 quantile-based bins
- bin_equal_frequency_5: Split into 5 quantile-based bins
- bin_equal_frequency_10: Split into 10 quantile-based bins
- bin_equal_width_2: Split into 2 equal-width intervals
- bin_equal_width_5: Split into 5 equal-width intervals
- bin_equal_width_10: Split into 10 equal-width intervals

Normalization operations:
- norm_min_max: Scale to [0, 1] range using (x - min) / (max - min)
- norm_log: Apply signed log1p transformation: sign(x) * log(1 + |x|)
- zscore_clip_3: Z-score normalization with clipping at ±3 standard deviations
- winsorize: Clip values to [5th percentile, 95th percentile] range
</available_transformations>

<constraints>
CRITICAL RULES:
1. You can apply AT MOST ONE binning operation per column
2. You can apply AT MOST ONE normalization operation per column
3. You CAN apply one binning AND one normalization to the same column (order matters)
4. Each column can receive 0, 1, or 2 transformations total
5. Do not apply transformations to the treatment or outcome variables
</constraints>

<output_format>
Return a JSON array of transformation steps in execution order:
[
  {"column": "feature_name", "operation": "operation_name"},
  {"column": "feature_name", "operation": "operation_name"}
]

If target ATE is already achieved: return empty array []
</output_format>
</task_description>"""

DO_NOT_THINK = "Do not include any text before or after the JSON array."
def create_compact_steering_prompt(df, current_ate, target_ate, epsilon,
                                   treatment_col, outcome_col, few_shots_prompt=""):
    """
    Minimal version - only essential statistics, no sample data.
    """

    covariate_cols = [c for c in df.columns if c not in [treatment_col, outcome_col]]

    # Create a compact stats dictionary
    stats = {}
    for col in covariate_cols:
        stats[col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std(),
            'median': df[col].median(),
            'skew': df[col].skew(),
            'p5': df[col].quantile(0.05),
            'p95': df[col].quantile(0.95),
            'corr_outcome': df[[col, outcome_col]].corr().iloc[0, 1],
            'corr_treatment': df[[col, treatment_col]].corr().iloc[0, 1]
        }

    prompt = f"""
Your goal is to steer the Average Treatment Effect (ATE) of a dataset toward a target value by selecting optimal data transformations.
{few_shots_prompt}
Dataset: {len(df):,} rows × {len(covariate_cols)} covariates
Treatment: {treatment_col} | Outcome: {outcome_col}

Current ATE: {current_ate:.6f}
Target ATE: {target_ate:.6f}
Epsilon: {epsilon:.6f}
Need to {"INCREASE" if target_ate > current_ate else "DECREASE"} ATE by {abs(target_ate - current_ate):.6f}

<feature_statistics>
"""

    # Compact table format
    for col, s in stats.items():
        prompt += f"\n{col}:"
        prompt += f"\n  Range: [{s['min']:.3f}, {s['max']:.3f}] | Mean: {s['mean']:.3f} ± {s['std']:.3f}"
        prompt += f"\n  Skew: {s['skew']:.2f} | P5-P95: [{s['p5']:.3f}, {s['p95']:.3f}]"
        prompt += f"\n  Corr(outcome): {s['corr_outcome']:+.3f} | Corr(treatment): {s['corr_treatment']:+.3f}"

    prompt += """
</feature_statistics>

Propose transformations to steer ATE to target. Return JSON array:
[{"column": "col_name", "operation": "op_name"}, ...]
"""

    return prompt


def create_few_shot_example(
        df: pd.DataFrame,
        current_ate: float,
        target_ate: float,
        epsilon: float,
        treatment_col: str,
        outcome_col: str,
        transformations: List[Dict[str, str]],
        explanation: str = ""
) -> str:
    """
    Create a few-shot example matching the compact prompt format.
    """

    covariate_cols = [c for c in df.columns if c not in [treatment_col, outcome_col]]

    # Create compact stats (same as main prompt)
    stats = {}
    for col in covariate_cols:
        stats[col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std(),
            'median': df[col].median(),
            'skew': df[col].skew(),
            'p5': df[col].quantile(0.05),
            'p95': df[col].quantile(0.95),
            'corr_outcome': df[[col, outcome_col]].corr().iloc[0, 1],
            'corr_treatment': df[[col, treatment_col]].corr().iloc[0, 1]
        }

    example = f"""
Dataset: {len(df):,} rows × {len(covariate_cols)} covariates
Treatment: {treatment_col} | Outcome: {outcome_col}

Current ATE: {current_ate:.6f}
Target ATE: {target_ate:.6f}
Epsilon: {epsilon:.6f}
Need to {"INCREASE" if target_ate > current_ate else "DECREASE"} ATE by {abs(target_ate - current_ate):.6f}

<feature_statistics>
"""

    # Compact table format (same as main prompt)
    for col, s in stats.items():
        example += f"\n{col}:"
        example += f"\n  Range: [{s['min']:.3f}, {s['max']:.3f}] | Mean: {s['mean']:.3f} ± {s['std']:.3f}"
        example += f"\n  Skew: {s['skew']:.2f} | P5-P95: [{s['p5']:.3f}, {s['p95']:.3f}]"
        example += f"\n  Corr(outcome): {s['corr_outcome']:+.3f} | Corr(treatment): {s['corr_treatment']:+.3f}"

    example += "\n</feature_statistics>\n"

    if explanation:
        example += f"\n<reasoning>\n{explanation}\n</reasoning>\n"

    example += f"\n<solution>\n{json.dumps(transformations, indent=2)}\n</solution>\n\n"

    return example


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


FEW_SHOT_EXAMPLE_TWINS = create_few_shot_example(twins_data_set, 0.06, 0.0019, 0.000001, 'treatment', 'outcome', [
    {'column': 'adequacy', 'operation': 'zscore_clip_3'},
    {'column': 'lung', 'operation': 'bin_equal_frequency_2'},
    {'column': 'wt', 'operation': 'bin_equal_frequency_2'}
])
FEW_SHOT_EXAMPLE_LALONDE = create_few_shot_example(lalonde_data_set, 1671, 1577, 1, 'treatment', 'outcome', [
    {'column': 'age', 'operation': 'bin_equal_width_5'},
    {'column': 'black', 'operation': 'bin_equal_frequency_2'},
    {'column': 'education', 'operation': 'bin_equal_frequency_2'},
    {'column': 'hispanic', 'operation': 'bin_equal_frequency_2'}
])

# if __name__ == '__main__':
#     # print(create_compact_steering_prompt(twins_data_set, 0.06, -0.06, 0.06,'treatment', 'outcome'))
#     few_shot_prompt = create_compact_steering_prompt(twins_data_set, 0.06, -0.06, 0.06,'treatment', 'outcome',create_few_shots_prompt([FEW_SHOT_EXAMPLE_TWINS, FEW_SHOT_EXAMPLE_LALONDE]))
#     print(few_shot_prompt)
