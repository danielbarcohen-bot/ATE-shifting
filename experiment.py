import concurrent
import random
from typing import List, Callable

import numpy as np
import pandas as pd

from prompts import SYSTEM_PROMPT_CLAUDE, create_compact_steering_prompt, create_few_shots_prompt, \
    FEW_SHOT_EXAMPLE_TWINS, FEW_SHOT_EXAMPLE_LALONDE, DO_NOT_THINK
from search_methods.LLM_search import LLMSearch
from search_methods.Random_search import RandomSearch
# from search_methods.AStar_search import AStarATESearch
from search_methods.brute_force_ATE_search import BruteForceATESearch
from search_methods.probe_ATE_search import ProbeATESearch
from search_methods.probe_ATE_search_heuristic_linear_reg import ProbeATESearchLinearRegHeuristic
from search_methods.probe_ATE_search_no_bit_mask import ProbeATESearchNoBitMask
from search_methods.probe_ATE_search_no_hash import ProbeATESearchNoHash
from search_methods.probe_ATE_search_no_lazy_eval import ProbeATESearchNoLazyEval
from search_methods.pruning_ATE_search import PruneATESearch
from utils import calculate_ate_linear_regression_lstsq


class Experiment:
    def __init__(
            self, df: pd.DataFrame, transformations_dict: dict[str, Callable], common_causes: List[str],
            target_ate: float, epsilon: float,
            max_length: int):
        self.df = df
        self.transformations_dict = transformations_dict
        self.common_causes = common_causes
        self.target_ate = target_ate
        self.epsilon = epsilon
        self.max_length = max_length

    def run_brute(self):
        return BruteForceATESearch().search(df=self.df, common_causes=self.common_causes, target_ate=self.target_ate,
                                            epsilon=self.epsilon,
                                            max_seq_length=self.max_length,
                                            transformations_dict=self.transformations_dict)

    def run_prune(self):
        return PruneATESearch().search(df=self.df, common_causes=self.common_causes, target_ate=self.target_ate,
                                       epsilon=self.epsilon,
                                       max_seq_length=self.max_length, transformations_dict=self.transformations_dict)

    # def run_parallel_prune(self):
    #     return PruneATESearch().search_parallel_f(df=self.df, common_causes=self.common_causes,
    #                                             target_ate=self.target_ate,
    #                                             epsilon=self.epsilon,
    #                                             max_seq_length=self.max_length)
    # def run_AStar(self):
    #     return AStarATESearch().search(df=self.df, common_causes=self.common_causes,
    #                                             target_ate=self.target_ate,
    #                                             epsilon=self.epsilon,
    #                                             max_seq_length=self.max_length)
    def run_probe(self):
        return ProbeATESearch().search(df=self.df, common_causes=self.common_causes,
                                       target_ate=self.target_ate,
                                       epsilon=self.epsilon,
                                       max_seq_length=self.max_length, transformations_dict=self.transformations_dict)

    # def run_probe_rows_sample(self, k):
    #     success_times = []
    #     fail_times = []
    #
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    #         futures = [executor.submit(ProbeATESearch().search, self.df.sample(frac=0.1 * int(k)), self.common_causes,self.target_ate,self.epsilon,self.max_length, self.transformations_dict) for _ in range(10)]
    #         for future in concurrent.futures.as_completed(futures):
    #             try:
    #                 res, run_time = future.result(timeout=RUN_TIMEOUT)
    #                 if res == "Search failed to find a solution within max steps.":
    #                     fail_times.append(run_time)
    #                 else:
    #                     success_times.append(run_time)
    #             except TimeoutError:
    #                 fail_times.append(run_time)
    #
    #     print(f"Success times: {success_times}")
    #     if success_times:
    #         print(f"Success times mean: {np.mean(success_times)}")
    #     print(f"Fail times: {fail_times}")
    #     if fail_times:
    #         print(f"Fail times mean: {np.mean(fail_times)}")
    #     return success_times, fail_times
    #
    # def run_probe_cols_sample(self, k):
    #     success_times = []
    #     fail_times = []
    #
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    #         # futures = [executor.submit(ProbeATESearch().search, self.df, random.sample(self.common_causes,k=k),self.target_ate,self.epsilon,self.max_length, self.transformations_dict) for _ in range(10)]
    #         futures = [executor.submit(ProbeATESearch().search, self.df[subset + ["treatment", "outcome"]], subset,
    #                                    self.target_ate, self.epsilon, self.max_length, self.transformations_dict) for
    #                    subset in [random.sample(self.common_causes, k=int(k)) for _ in range(10)]]
    #
    #         for future in concurrent.futures.as_completed(futures):
    #             res, run_time = future.result()
    #             if res == "Search failed to find a solution within max steps.":
    #                 fail_times.append(run_time)
    #             else:
    #                 success_times.append(run_time)
    #     print(f"Success times: {success_times}")
    #     if success_times:
    #         print(f"Success times mean: {np.mean(success_times)}")
    #     print(f"Fail times: {fail_times}")
    #     if fail_times:
    #         print(f"Fail times mean: {np.mean(fail_times)}")
    #     return success_times, fail_times


    def run_probe_line_reg_heuristic(self):
        return ProbeATESearchLinearRegHeuristic().search(df=self.df, common_causes=self.common_causes,
                                                         target_ate=self.target_ate,
                                                         epsilon=self.epsilon,
                                                         max_seq_length=self.max_length,
                                                         transformations_dict=self.transformations_dict)

    def run_probe_no_hash(self):
        return ProbeATESearchNoHash().search(df=self.df, common_causes=self.common_causes,
                                             target_ate=self.target_ate,
                                             epsilon=self.epsilon,
                                             max_seq_length=self.max_length,
                                             transformations_dict=self.transformations_dict)

    def run_probe_no_bit_mask(self):
        return ProbeATESearchNoBitMask().search(df=self.df, common_causes=self.common_causes,
                                                target_ate=self.target_ate,
                                                epsilon=self.epsilon,
                                                max_seq_length=self.max_length,
                                                transformations_dict=self.transformations_dict)

    def run_probe_no_lazy_eval(self):
        return ProbeATESearchNoLazyEval().search(df=self.df, common_causes=self.common_causes,
                                                 target_ate=self.target_ate,
                                                 epsilon=self.epsilon,
                                                 max_seq_length=self.max_length,
                                                 transformations_dict=self.transformations_dict)
    def run_llm_zero_shot(self, with_COT = False):
        curr_ate = calculate_ate_linear_regression_lstsq(self.df, 'treatment', 'outcome', self.common_causes)
        prompt = create_compact_steering_prompt(self.df, curr_ate, self.target_ate, self.epsilon, 'treatment', 'outcome')
        if not with_COT:
            prompt = prompt + DO_NOT_THINK
        return LLMSearch(SYSTEM_PROMPT_CLAUDE, prompt).search(df=self.df, common_causes=self.common_causes,
                                                 target_ate=self.target_ate,
                                                 epsilon=self.epsilon,
                                                 max_seq_length=self.max_length,
                                                 transformations_dict=self.transformations_dict)
    def run_llm_few_shot(self, with_COT = False):
        curr_ate = calculate_ate_linear_regression_lstsq(self.df, 'treatment', 'outcome', self.common_causes,)
        prompt = create_compact_steering_prompt(self.df, curr_ate, self.target_ate, self.epsilon, 'treatment',
                                                'outcome', create_few_shots_prompt([FEW_SHOT_EXAMPLE_TWINS, FEW_SHOT_EXAMPLE_LALONDE]))
        if not with_COT:
            prompt = prompt + DO_NOT_THINK
        return LLMSearch(SYSTEM_PROMPT_CLAUDE, prompt).search(df=self.df, common_causes=self.common_causes,
                                                 target_ate=self.target_ate,
                                                 epsilon=self.epsilon,
                                                 max_seq_length=self.max_length,
                                                 transformations_dict=self.transformations_dict)


class RandomExperiment:
    def __init__(self, df: pd.DataFrame, transformations_dict: dict[str, Callable], common_causes: List[str],
                 target_ate: float, epsilon: float,
                 sequence_length: int):
        self.df = df
        self.transformations_dict = transformations_dict
        self.common_causes = common_causes
        self.target_ate = target_ate
        self.epsilon = epsilon
        self.sequence_length = sequence_length

    def run_random(self):
        ates = []
        for _ in range(10):
            seq, ate = RandomSearch().search(df=self.df, transformations_dict=self.transformations_dict,
                                             common_causes=self.common_causes, sequence_length=self.sequence_length)
            ates.append(ate.item())
            if abs(ate - self.target_ate) < self.epsilon:
                print(f"found solution, ATE is {ate}, sequence is \n{seq}")
        print(f"ATEs are {sorted(ates)}")
        distances = sorted([abs(ate - self.target_ate) - self.epsilon for ate in ates])
        print(f"distances from target:\n{distances}")
        print(f"avg distance from target:\n{sum(distances) / len(distances)}")
