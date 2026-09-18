import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments import largest_data_transformations, LEGAL_OPS_BY_TYPE
from search_methods.OE_ATE_search import OEATESearch
from search_methods.Random_search import RandomSearch
from utils import analyze_ate_search_space


def plot_ate_analysis_interactive(summary_df, dataset_name):
    summary_df = summary_df.copy()

    # Midpoint of interval

    if pd.api.types.is_string_dtype(summary_df['bucket_range']):
        # Extract all numbers (including decimals/negatives) from the string
        # e.g., "(1566.431, 1567.082]" -> [1566.431, 1567.082]
        extracted_bounds = summary_df['bucket_range'].str.findall(r"[-+]?\d*\.\d+|\d+")

        # Calculate the midpoint by averaging the left and right bounds
        summary_df['x_coord'] = extracted_bounds.apply(
            lambda x: (float(x[0]) + float(x[1])) / 2 if isinstance(x, list) and len(x) == 2 else None)
    else:
        # If it is already a proper Interval object, pull the midpoint directly
        summary_df['x_coord'] = summary_df['bucket_range'].apply(lambda x: x.mid if pd.notna(x) else None)

    # Automatic width from spacing
    x = np.sort(summary_df['x_coord'].values)

    if len(x) > 1:
        width = np.min(np.diff(x)) * 0.8
    else:
        width = 0.1

    # Figure
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bars
    ax1.bar(
        summary_df['x_coord'],
        summary_df['count'],
        width=width,
        color='darkgreen',
        alpha=0.9
    )

    ax1.set_xlabel('ATE Bucket Range')
    ax1.set_ylabel('Number of Sequences', color='darkgreen')
    ax1.set_yscale('log')

    # Second axis
    ax2 = ax1.twinx()

    ax2.plot(
        summary_df['x_coord'],
        summary_df['min_length'],
        color='red',
        linewidth=2,
        marker='o',
        alpha=0.3
    )

    ax2.set_ylabel('Min Sequence Length', color='red')

    plt.title(
        f'({dataset_name}) Causal Search Space Analysis: '
        f'ATE Density vs. Pipeline Complexity'
    )

    plt.tight_layout()
    plt.show()


def add_random_walks(df: pd.DataFrame, common_causes, transformations_dict, seq_ates_arr, legal_ops_by_type, num_iterations=2000):
    start_time = time.time()
    # 1. Convert to dictionary for O(1) lightning-fast lookup
    search_space_registry = {sequence: ate for sequence, ate in seq_ates_arr}

    initial_count = len(search_space_registry)
    print(f"Initialized registry with {initial_count} existing unique paths.")

    # 2. Loop through requested iterations
    for i in range(num_iterations):
        if time.time() - start_time > 1800:  # max 30 minutes of generating
            break
        sequence_length = np.random.randint(1, 25)
        # Call your generator to build a pipeline and compute ATE
        sequence, ate = RandomSearch().search(df, common_causes, transformations_dict, sequence_length, legal_ops_by_type)

        # Ensure sequence is immutable (tuple) so it can be hashed
        sequence_tuple = tuple(sequence)

        # 3. Deduplication check
        if sequence_tuple not in search_space_registry:
            search_space_registry[sequence_tuple] = ate

    # 4. Format back into your required list of tuples structure
    updated_results = list(search_space_registry.items())

    new_paths_found = len(updated_results) - initial_count
    print(f"Finished! Found {new_paths_found} brand-new unique paths.")
    print(f"Total search space registry now stands at {len(updated_results)} routes.")

    return updated_results


def get_ate_bins_df(df, common_causes, time_out_sec, df_name):
    oe_search_result = OEATESearch().search(df=df, common_causes=common_causes, target_ate=np.inf,
                                            epsilon=0, transformations_dict=largest_data_transformations,
                                            time_out_sec=time_out_sec, legal_ops_by_type=LEGAL_OPS_BY_TYPE)

    save_tuples_to_csv(f"search_space_OE_{df_name}.csv", oe_search_result['seq_ates'])

    seq_ates_with_random = add_random_walks(df, common_causes, largest_data_transformations,
                                            oe_search_result['seq_ates'], LEGAL_OPS_BY_TYPE)

    save_tuples_to_csv(f"search_space_total_{df_name}.csv", seq_ates_with_random)

    ate_bins_data = analyze_ate_search_space(seq_ates_with_random)
    return ate_bins_data


def save_tuples_to_csv(file_name, data):
    if not file_name.endswith('.csv'):
        file_name += '.csv'

    with open(file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(data)
