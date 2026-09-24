import csv
import io
import sys

from data_loader import data_factory
from experiments import largest_data_transformations, LEGAL_OPS_BY_TYPE
from specific_experiments.Q1.search_space_utils import add_random_walks, load_tuples_from_csv
from utils import analyze_ate_search_space

# Second half of get_ate_bins_df: extend an existing OE search space file with random walks, then bin it.
if __name__ == '__main__':
    dataset = sys.argv[1]
    num_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    df = data_factory(dataset)
    common_causes = df.columns.difference(["treatment", "outcome"], sort=False).tolist()
    print(common_causes)

    # pre-populated by the OE search (get_ate_bins_df)
    seq_ates_path = f"search_space_OE_{dataset}.csv"
    seen_sequences = [sequence for sequence, _ in load_tuples_from_csv(seq_ates_path)]

    raw = io.BufferedWriter(io.FileIO(seq_ates_path, "a"))
    with io.TextIOWrapper(raw, encoding="utf-8", newline="", write_through=True) as out:
        seq_ates_writer = csv.writer(out)
        whole_table_ops = ['isolationForest', 'drop_duplicates']
        add_random_walks(df, common_causes, largest_data_transformations, whole_table_ops, seq_ates_writer, seen_sequences,
                         LEGAL_OPS_BY_TYPE, num_iterations=num_iterations)

    bins = analyze_ate_search_space(load_tuples_from_csv(seq_ates_path))
    bins.to_csv(f"ate_bins_data_{dataset}.csv", index=False)
