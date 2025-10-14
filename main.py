from data_loader import TwinsDataLoader
from search_methods.brute_force_ATE_search import BruteForceATESearch
from search_methods.pruning_ATE_search import PruneATESearch

if __name__ == '__main__':
    common_causes = ['wt', 'hydra', 'nprevistq', 'gestat10']
    target_ate = 0.005
    epsilon = 0.0001
    max_length = 5
    print("start")
    df = TwinsDataLoader().load_data()
    print("aded")
    BruteForceATESearch().search(df=df, common_causes=common_causes, target_ate=target_ate, epsilon=epsilon,
                                 max_seq_length=max_length)
    PruneATESearch().search(df=df, common_causes=common_causes, target_ate=target_ate, epsilon=epsilon,
                            max_seq_length=max_length)
