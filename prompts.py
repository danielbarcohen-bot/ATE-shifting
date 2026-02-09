# SYSTEM_PROMPT = """
# <role>
# You are an expert Causal Inference Optimization Engine. Your goal is to steer the Average Treatment Effect (ATE) of a linear regression model towards a specific target by applying data transformations to covariates.
# </role>
#
# <objective>
# Analyze the provided dataset summary, current ATE, and regression context. Propose a sequence of strictly allowed data transformations to move the ATE into the range [Target - Epsilon, Target + Epsilon].
# </objective>
#
# <constraints>
# 1. **Single Group Per Column:** You strictly cannot apply more than one operation from the same IDENTITY GROUP to the same column. (e.g., if you apply 'bin_equal_width_5' to 'age', you cannot apply 'bin_equal_frequency_10' to 'age' later).
# 2. **Sequential Logic:** Operations are applied in the order you list them.
# 3. **valid JSON:** Output must be valid JSON format only.
# </constraints>
#
# <allowed_transformations>
#     <group id="A_DISCRETIZATION">
#         bin_equal_frequency_2, bin_equal_frequency_5, bin_equal_frequency_10
#         bin_equal_width_2, bin_equal_width_5, bin_equal_width_10
#     </group>
#
#     <group id="B_NORMALIZATION">
#         min_max_norm,      log_norm           </group>
#
#     <group id="C_OUTLIER_HANDLING">
#         zscore_clip_3,     winsorize          </group>
# </allowed_transformations>
#
# <output_format>
# Return a single JSON object. Do not include markdown formatting like ```json.
#
# {
#   "sequence": [
#     {
#       "step_id": 1,
#       "column": "column_name",
#       "operation": "operation_name_from_list",
#       "group_id": "A_DISCRETIZATION"
#     }
#   ]
# }
# </output_format>
# """
#
# USER_PROMPT = """
# <task>
# Generate a sequence of data transformations to steer the Average Treatment Effect (ATE) of a linear regression towards a TARGET_ATE.
# </task>
#
# <data_context>
# - Current ATE: {current_ate}
# - Target ATE: {target_ate}
# - Epsilon: {epsilon}
# - Treatment Column: {treatment_col}
# - Outcome Column: {outcome_col}
# - Data Statistics:
# {data_stats}
# </data_context>
#
# <transformation_rules>
# You must select operations from these groups.
# CRITICAL: You may apply ONLY ONE operation per group to any single column.
#
# Group A (Binning):
# bin_equal_frequency_2, bin_equal_frequency_5, bin_equal_frequency_10, bin_equal_width_2, bin_equal_width_5, bin_equal_width_10
#
# Group B (Normalization):
# min_max_norm, log_norm
#
# Group C (Outlier Handling):
# zscore_clip_3, winsorize
# </transformation_rules>
#
# <constraint>
# Return ONLY a valid JSON object. No preamble, no explanation, no Chain-of-Thought, and no markdown formatting.
# </constraint>
#
# <output_schema>
# {
#   "sequence": [
#     {"column": "string", "operation": "string"}
#   ]
# }
# </output_schema>
# """

from data_loader import TwinsDataLoader
twins_data_set = TwinsDataLoader().load_data().dropna()
SYSTEM_PROMPT_gemini = """
You are a Causal Inference Engine. Your role is to generate precise data transformation sequences to steer ATE. 
Output ONLY valid JSON matching the user's schema. No conversation.
"""

SYSTEM_PROMPT_claude = """<task_description>
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
- min_max_norm: Scale to [0, 1] range using (x - min) / (max - min)
- log_norm: Apply signed log1p transformation: sign(x) * log(1 + |x|)
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
def generate_user_prompt_gemini(df, current_ate, target_ate, epsilon, data_stats, few_shot_example=""):
    user_prompt = f"""
    <mission>
    Steer the Average Treatment Effect (ATE) of the provided dataset toward the TARGET_ATE within the EPSILON range.
    - Current ATE: {current_ate}
    - Target ATE: {target_ate}
    - Epsilon: {epsilon}
    - Treatment: treatment
    - Outcome: outcome
    </mission>
    
    <data_summary>
    {data_stats}
    </data_summary>
    <raw_data_sample>
    {df.head(10).to_csv(index=False)}
    </raw_data_sample>

    
    <transformation_logic>
    You must select operations from the following IDENTITY GROUPS. 
    RULE: You cannot apply more than one operation from the same GROUP to a single column.
    
    ### GROUP A: Discretization (Non-Linearity Handling)
    - bin_equal_frequency_[k]: Quantile-based binning. Divides data into k bins such that each bin has roughly the same number of observations. Useful for handling skewed densities.
    - bin_equal_width_[k]: Interval-based binning. Divides the range of data into k equal-sized numerical intervals. Useful for uniform value distribution.
    * note - k can be [2, 5, 10]
    
    ### GROUP B: Normalization (Scale Adjustment)
    - min_max_norm: Rescales the range to [0, 1].
    - log_norm: Performs 'Signed log1p' [sign(x) * ln(1 + |x|)]. This handles heavy-tailed distributions while preserving the original sign of the data.
    
    ### GROUP C: Outlier Mitigation (Variance Reduction)
    - zscore_clip_3: Calculates the Z-score and caps all values at +/- 3 standard deviations from the mean.
    - winsorize: Clips extreme values to the 5th and 95th percentile respectively, replacing outliers with the nearest "safe" values.
    </transformation_logic>
    {few_shot_example}
    <output_schema>
    Return a JSON object with a "sequence" key containing a list of objects with "column" and "operation".
    Example: { "sequence": [{"column": "age", "operation": "winsorize"}] }
    </output_schema>
    """
    return user_prompt

FEW_SHOT_EXAMPLE = f"""

<examples>
    <example>
        <input>
        - Current ATE: 0.06
        - Target ATE: 0.0019
        - Epsilon: 0.000001
        - Treatment: treatment
        - Outcome: outcome
        
        <data_summary>
        {twins_data_set.describe().to_csv()}
        </data_summary>
        <raw_data_sample>
        {twins_data_set.head(10).to_csv(index=False)}
        </raw_data_sample>

        </input>
        <output>
        {{ "sequence": [{{"column": "adequacy", "operation": "zscore_clip_3"}}, {{"column": "lung", "operation": "bin_equal_frequency_2"}}, {{"column": "wt", "bin_equal_frequency_2": "winsorize"}}] }}
        </output>
    </example>
</examples>

"""

import pandas as pd
import numpy as np


def create_steering_prompt(df, current_ate, target_ate, epsilon,
                           treatment_col, outcome_col, sample_size=50):
    """
    Create a prompt optimized for large dataframes.
    Only sends statistical summaries + small sample, not full data.
    """

    # Get covariate columns (exclude treatment and outcome)
    covariate_cols = [c for c in df.columns if c not in [treatment_col, outcome_col]]

    # Calculate correlations
    corr_with_outcome = df[covariate_cols + [outcome_col]].corr()[outcome_col].drop(outcome_col)
    corr_with_treatment = df[covariate_cols + [treatment_col]].corr()[treatment_col].drop(treatment_col)

    # Get distribution statistics
    stats_df = df[covariate_cols].describe(percentiles=[.05, .25, .5, .75, .95])

    # Add skewness and kurtosis
    skew_series = df[covariate_cols].skew()
    kurt_series = df[covariate_cols].kurtosis()

    # Get a stratified sample (if treatment is binary, sample from both groups)
    if df[treatment_col].nunique() == 2:
        sample_df = df.groupby(treatment_col, group_keys=False).apply(
            lambda x: x.sample(min(len(x), sample_size // 2), random_state=42)
        )
    else:
        sample_df = df.sample(min(len(df), sample_size), random_state=42)

    prompt = f"""
CAUSAL INFERENCE STEERING TASK

<dataset_info>
Total rows: {len(df):,}
Total columns: {len(df.columns)}
Treatment variable: {treatment_col}
Outcome variable: {outcome_col}
Covariate columns ({len(covariate_cols)}): {', '.join(covariate_cols)}
</dataset_info>

<current_state>
Current ATE: {current_ate:.6f}
Target ATE: {target_ate:.6f}
Epsilon: {epsilon:.6f}
Acceptable range: [{target_ate - epsilon:.6f}, {target_ate + epsilon:.6f}]
Gap to target: {abs(target_ate - current_ate):.6f}
Direction needed: {"INCREASE" if target_ate > current_ate else "DECREASE"}
</current_state>

<distribution_statistics>
{stats_df.to_string()}

Skewness by column:
{skew_series.to_string()}

Kurtosis by column:
{kurt_series.to_string()}
</distribution_statistics>

<correlation_analysis>
Correlation with OUTCOME ({outcome_col}):
{corr_with_outcome.sort_values(ascending=False).to_string()}

Correlation with TREATMENT ({treatment_col}):
{corr_with_treatment.sort_values(ascending=False).to_string()}
</correlation_analysis>

<sample_data>
Representative sample ({len(sample_df)} rows):
{sample_df.to_csv(index=False, float_format='%.4f')}
</sample_data>

<analysis_guidance>
Key indicators to consider:
1. High skewness (|skew| > 1) → Consider log_norm or winsorize
2. Heavy tails (kurtosis > 3) → Consider winsorize or zscore_clip_3
3. Wide ranges → Consider min_max_norm or binning
4. Strong correlations → Prioritize these features for transformation
5. Nonlinear relationships → Consider binning strategies

Current ATE is {current_ate:.6f}, target is {target_ate:.6f}.
You need to {"INCREASE" if target_ate > current_ate else "DECREASE"} the ATE by {abs(target_ate - current_ate):.6f}.
</analysis_guidance>

Analyze the statistics and propose an optimal sequence of transformations to steer the ATE toward the target.

Think through:
1. Which features are most influential (high correlation with outcome/treatment)?
2. Which features have problematic distributions that affect ATE estimation?
3. What transformations will strategically shift the ATE in the needed direction?
4. What is the optimal order of operations?

Return ONLY a JSON array of transformations.
"""

    return prompt


# Alternative: Even more compact version for VERY large datasets
def create_compact_steering_prompt(df, current_ate, target_ate, epsilon,
                                   treatment_col, outcome_col):
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
CAUSAL INFERENCE STEERING TASK

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

if __name__ == '__main__':
    print(create_compact_steering_prompt(twins_data_set, 0.06, -0.06, 0.06,'treatment', 'outcome'))