import argparse
import sys

from data_loader import data_factory
from specific_experiments.Q1.search_space_utils import get_ate_bins_df

if __name__ == '__main__':
    # Unpack args:
    # parser = argparse.ArgumentParser()
    # parser.add_argument('dataset',type=str)
    # args = parser.parse_args()
    dataset = sys.argv[2]

    df = data_factory(dataset)
    common_causes = df.columns.difference(["treatment", "outcome"], sort=False).tolist()
    print(common_causes)

    MAX_RUN_TIME_SEC = 120# 60 * 5  # 18000
    bins = get_ate_bins_df(df, common_causes, MAX_RUN_TIME_SEC, dataset,100)
    bins.to_csv(f"ate_bins_data_{dataset}.csv", index=False)