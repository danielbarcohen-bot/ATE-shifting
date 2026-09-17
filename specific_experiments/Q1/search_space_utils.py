import ast
import csv
import io
import time
import random

import numpy as np
import pandas as pd

from experiments import largest_data_transformations, LEGAL_OPS_BY_TYPE, LEGAL_FILL_BY_TYPE
from search_methods.OE_ATE_search import OEATESearch
from search_methods.Random_search import RandomSearch
from utils import analyze_ate_search_space


def plot_ate_analysis_interactive(summary_df, dataset_name):
    import matplotlib.pyplot as plt  # local import to not mess up importing the module
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


def add_random_walks(df: pd.DataFrame,
                     common_causes,
                     transformations_dict,
                     whole_table_ops,
                     seq_ates_writer,
                     seen_sequences: list,
                     legal_ops_by_type, num_iterations=2000):
    start_time = time.time()
    # 1. Only the sequences are kept in memory (for dedup); new (sequence, ATE) pairs go to seq_ates_writer
    # fill_len = len(list(filter(lambda y: y[0].startswith('fill'), next(filter(lambda x: x[0][0].startswith('fill_'), seen_sequences)))))
    min_gen_len = max(len(list(filter(lambda x: not x[0].startswith('fill_') , seq))) for seq in seen_sequences)

    initial_count = len(seen_sequences)
    print(f"Initialized registry with {initial_count} existing unique paths.")
    new_paths_found = 0

    # 2. Loop through requested iterations
    for i in range(num_iterations):
        if time.time() - start_time > 1800:  # max 30 minutes of generating
            break
        sequence_length = np.random.randint(min_gen_len, 25)
        to_extend = random.choice(seen_sequences)
        if len(to_extend) == sequence_length:
            sequence_length += 1
        # Call your generator to build a pipeline and compute ATE
        print(f"search {i}")
        sequence, ate = RandomSearch().search(
            df, common_causes,
            transformations_dict,
            whole_table_ops,
            sequence_length,
            legal_ops_by_type,
            to_extend
        )

        # Ensure sequence is immutable (tuple) so it can be hashed
        sequence_tuple = tuple(sequence)

        # 3. Deduplication check
        if sequence_tuple not in seen_sequences:
            # seen_sequences.add(sequence_tuple)
            seq_ates_writer.writerow((sequence_tuple, ate))
            new_paths_found += 1

    # new_paths_found = len(seen_sequences) - initial_count
    print(f"Finished! Explored {new_paths_found} more programs.")
    print(f"Total search space registry now stands at {len(seen_sequences)} routes.")


def get_ate_bins_df(df, common_causes, time_out_sec, df_name, add_random: int = 2000, seq_ates_buffer_bytes: int = 1 << 20):
    seq_ates_path = f"search_space_OE_{df_name}.csv"
    raw = io.BufferedWriter(io.FileIO(seq_ates_path, "w"), buffer_size=seq_ates_buffer_bytes)
    with io.TextIOWrapper(raw, encoding="utf-8", newline="", write_through=True) as out:
        seq_ates_writer = csv.writer(out)
        whole_table_ops = ['isolationForest', 'drop_duplicates']
        OEATESearch().search(df=df, common_causes=common_causes, target_ate=np.inf,
                             epsilon=0,
                             transformations_dict=largest_data_transformations,
                             whole_table_ops=whole_table_ops,
                             time_out_sec=time_out_sec,
                             ops_by_type=LEGAL_OPS_BY_TYPE,
                             fill_by_type=LEGAL_FILL_BY_TYPE,
                             seq_ates_writer=seq_ates_writer)

        # flush so the OE rows are on disk before reading back the sequences seen so far
        out.flush()
        seen_sequences = [sequence for sequence, _ in load_tuples_from_csv(seq_ates_path)]
        add_random_walks(df, common_causes, largest_data_transformations, whole_table_ops, seq_ates_writer, seen_sequences,
                         LEGAL_OPS_BY_TYPE,num_iterations=add_random)

    ate_bins_data = analyze_ate_search_space(load_tuples_from_csv(seq_ates_path))
    return ate_bins_data


def save_tuples_to_csv(file_name, data):
    if not file_name.endswith('.csv'):
        file_name += '.csv'

    with open(file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(data)


def load_tuples_from_csv(file_name):
    # sequences are stored as tuple reprs; literal_eval turns them back into hashable tuples
    with open(file_name, newline="", encoding="utf-8") as f:
        return [(ast.literal_eval(seq), float(ate)) for seq, ate in csv.reader(f)]
