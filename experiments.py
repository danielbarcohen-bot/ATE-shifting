from data_loader import TwinsDataLoader
from experiment import Experiment
from search_methods.brute_force_ATE_search import BruteForceATESearch
from search_methods.pruning_ATE_search import PruneATESearch

df = TwinsDataLoader().load_data()
common_causes = ['wt', 'hydra', 'nprevistq', 'gestat10']
target_ate = 0.005
epsilon = 0.0001
max_length = 5
experiment_1_example_and_poc = Experiment(df=df, common_causes=common_causes, target_ate=target_ate, epsilon=epsilon,
                                          max_length=max_length)

df = TwinsDataLoader().load_data()
common_causes = df.columns.difference(["treatment", "outcome"]).tolist()
target_ate = -1
epsilon = 0.5
max_length = 7
experiment_2_full_table = Experiment(df=df, common_causes=common_causes, target_ate=target_ate, epsilon=epsilon,
                                     max_length=max_length)

df = TwinsDataLoader().load_data()

EXPERIMENTS = {
    "EXP1": {  # poc example - shift ATE
        "df": df,
        "common_causes": ['wt', 'hydra', 'nprevistq', 'gestat10'],
        "target_ate": 0.005,
        "epsilon": 0.0001,
        "max_length": 5
    },
    "EXP2": {  # run algorithms on the whole dataset
        "df": df,
        "common_causes": df.columns.difference(["treatment", "outcome"]).tolist(),
        "target_ate": 0.005,
        "epsilon": 0.0001,
        "max_length": 5
    }
}
