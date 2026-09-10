from typing import List, Callable

import numpy as np
import pandas as pd

from F_size_experiments import run_F_experiment
from prompts import SYSTEM_PROMPT_CLAUDE, create_compact_steering_prompt, create_few_shots_prompt, \
    FEW_SHOT_EXAMPLE_TWINS, FEW_SHOT_EXAMPLE_LALONDE, DO_NOT_THINK
from search_methods.LLM_search import LLMSearch
# from search_methods.probe_ATE_search_heuristic_linear_reg import ProbeATESearchLinearRegHeuristic
from search_methods.OE_ATE_search_no_bit_mask import OEATESearchNoBitMask
from search_methods.OE_ATE_search_no_hash import OEATESearchNoHash
from search_methods.Random_search import RandomSearch
# from search_methods.AStar_search import AStarATESearch
from search_methods.brute_force_ATE_search import BruteForceATESearch
from search_methods.greedy_ATE_search import GreedyATESearch
from search_methods.probe_ATE_search import ProbeATESearch
from utils import calculate_ate_linear_regression_lstsq


class Experiment:
    def __init__(
            self, df: pd.DataFrame, transformations_dict: dict[str, Callable], common_causes: List[str],
            target_ate: float, epsilon: float,
            max_length: int, whole_df_ops: list[str] = None, legal_ops_by_type=None):
        self.df = df
        self.transformations_dict = transformations_dict
        self.common_causes = common_causes
        self.target_ate = target_ate
        self.epsilon = epsilon
        self.max_length = max_length
        self.whole_df_ops = whole_df_ops
        self.legal_ops_by_type = legal_ops_by_type

    def run_brute(self):
        return BruteForceATESearch().search(df=self.df, common_causes=self.common_causes, target_ate=self.target_ate,
                                            epsilon=self.epsilon,
                                            transformations_dict=self.transformations_dict,
                                            whole_df_ops=self.whole_df_ops)

    def run_prune(self):
        return ProbeATESearch(use_restart=False).search(df=self.df, common_causes=self.common_causes,
                                                        target_ate=self.target_ate,
                                                        epsilon=self.epsilon,
                                                        transformations_dict=self.transformations_dict,
                                                        whole_df_ops=self.whole_df_ops, legal_ops_by_type=self.legal_ops_by_type)

    # def run_AStar(self):
    #     return AStarATESearch().search(df=self.df, common_causes=self.common_causes,
    #                                             target_ate=self.target_ate,
    #                                             epsilon=self.epsilon,
    #                                             max_seq_length=self.max_length)
    def run_probe(self):
        return ProbeATESearch().search(df=self.df, common_causes=self.common_causes,
                                       target_ate=self.target_ate,
                                       epsilon=self.epsilon,
                                       transformations_dict=self.transformations_dict,
                                       whole_df_ops=self.whole_df_ops, legal_ops_by_type=self.legal_ops_by_type)

    def run_probe_no_hash(self):
        return ProbeATESearch(use_hash=False).search(df=self.df, common_causes=self.common_causes,
                                                     target_ate=self.target_ate,
                                                     epsilon=self.epsilon,
                                                     transformations_dict=self.transformations_dict,
                                                     whole_df_ops=self.whole_df_ops, legal_ops_by_type=self.legal_ops_by_type)

    # def run_prune_no_hash(self):
    #     return OEATESearchNoHash().search(df=self.df, common_causes=self.common_causes,
    #                                       target_ate=self.target_ate,
    #                                       epsilon=self.epsilon,
    #                                       max_seq_length=self.max_length,
    #                                       transformations_dict=self.transformations_dict)

    # def run_prune_no_bit_mask(self):
    #     return OEATESearchNoBitMask().search(df=self.df, common_causes=self.common_causes,
    #                                          target_ate=self.target_ate,
    #                                          epsilon=self.epsilon,
    #                                          max_seq_length=self.max_length,
    #                                          transformations_dict=self.transformations_dict)

    def run_llm_zero_shot(self, with_COT=False):
        curr_ate = calculate_ate_linear_regression_lstsq(self.df, 'treatment', 'outcome', self.common_causes)
        prompt = create_compact_steering_prompt(self.df, curr_ate, self.target_ate, self.epsilon, 'treatment',
                                                'outcome')
        if not with_COT:
            prompt = prompt + DO_NOT_THINK
        return LLMSearch(SYSTEM_PROMPT_CLAUDE, prompt).search(df=self.df, common_causes=self.common_causes,
                                                              target_ate=self.target_ate,
                                                              epsilon=self.epsilon,
                                                              max_seq_length=self.max_length,
                                                              transformations_dict=self.transformations_dict)

    def run_llm_few_shot(self, with_COT=False):
        curr_ate = calculate_ate_linear_regression_lstsq(self.df, 'treatment', 'outcome', self.common_causes, )
        prompt = create_compact_steering_prompt(self.df, curr_ate, self.target_ate, self.epsilon, 'treatment',
                                                'outcome', create_few_shots_prompt(
                [FEW_SHOT_EXAMPLE_TWINS, FEW_SHOT_EXAMPLE_LALONDE]))
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
            print(seq)
            ates.append(ate.item())
            if abs(ate - self.target_ate) < self.epsilon:
                print(f"found solution, ATE is {ate}, sequence is \n{seq}")
        print(f"ATEs are {sorted(ates)}")
        distances = sorted([abs(ate - self.target_ate) - self.epsilon for ate in ates])
        print(f"distances from target:\n{distances}")
        print(f"avg distance from target:\n{sum(distances) / len(distances)}")
        print(f"avg ATE is {np.mean(ates)}")


class ProbabilitiesExperiment(Experiment):
    def __init__(self, df: pd.DataFrame, transformations_dict: dict[str, Callable], common_causes: List[str],
                 target_ate: float, epsilon: float, max_sequence_length: int, op_probs: dict[str, float],
                 whole_df_ops: list[str] = None):
        super().__init__(df, transformations_dict, common_causes, target_ate, epsilon, max_sequence_length,
                         whole_df_ops)
        self.op_probs = op_probs

    def run_probe_no_restart(self):
        return ProbeATESearch(use_restart=False, op_probs=self.op_probs).search(df=self.df,
                                                                                common_causes=self.common_causes,
                                                                                target_ate=self.target_ate,
                                                                                epsilon=self.epsilon,
                                                                                transformations_dict=self.transformations_dict,
                                                                                whole_df_ops=self.whole_df_ops)

    def run_probe_no_restart_no_hash(self):
        return ProbeATESearch(use_restart=False, op_probs=self.op_probs, use_hash=False).search(df=self.df,
                                                                                                common_causes=self.common_causes,
                                                                                                target_ate=self.target_ate,
                                                                                                epsilon=self.epsilon,
                                                                                                transformations_dict=self.transformations_dict,
                                                                                                whole_df_ops=self.whole_df_ops)

    def run_probe(self):
        return ProbeATESearch(op_probs=self.op_probs).search(df=self.df, common_causes=self.common_causes,
                                                             target_ate=self.target_ate,
                                                             epsilon=self.epsilon,
                                                             transformations_dict=self.transformations_dict,
                                                             whole_df_ops=self.whole_df_ops)

    def run_probe_no_hash(self):
        return ProbeATESearch(op_probs=self.op_probs, use_hash=False).search(df=self.df,
                                                                             common_causes=self.common_causes,
                                                                             target_ate=self.target_ate,
                                                                             epsilon=self.epsilon,
                                                                             transformations_dict=self.transformations_dict,
                                                                             whole_df_ops=self.whole_df_ops)

    def run_probe_brute(self):
        return ProbeATESearch(op_probs=self.op_probs, use_restart=False, is_brute=True).search(df=self.df,
                                                                                               common_causes=self.common_causes,
                                                                                               target_ate=self.target_ate,
                                                                                               epsilon=self.epsilon,
                                                                                               transformations_dict=self.transformations_dict,
                                                                                               whole_df_ops=self.whole_df_ops)

    def run_greedy(self):
        return GreedyATESearch(self.op_probs, self.max_length).search(df=self.df, common_causes=self.common_causes,
                                                     target_ate=self.target_ate,
                                                     epsilon=self.epsilon,
                                                     transformations_dict=self.transformations_dict,
                                                     whole_df_ops=self.whole_df_ops)

    def run_brute_prob(self):
        return BruteForceATESearch(self.op_probs).search(df=self.df, common_causes=self.common_causes,
                                                         target_ate=self.target_ate,
                                                         epsilon=self.epsilon,
                                                         transformations_dict=self.transformations_dict)


class FExperiment(Experiment):
    def __init__(self, df: pd.DataFrame, transformations_dict: dict[str, Callable], common_causes: List[str],
                 target_ate: float, epsilon: float, max_sequence_length: int, op_probs: dict[str, float], i: int,
                 solution_sequence, whole_df_ops: list[str] = None):
        super().__init__(df, transformations_dict, common_causes, target_ate, epsilon, max_sequence_length,
                         whole_df_ops=whole_df_ops)
        self.op_probs = op_probs
        self.i = i
        self.solution_sequence = solution_sequence

    def run_probe_uniform(self):
        search_alg = ProbeATESearch()
        return run_F_experiment(
            search_algorithm=search_alg,
            df=self.df,
            common_causes=self.common_causes,
            target_ate=self.target_ate,
            epsilon=self.epsilon,
            transformations_dict=self.transformations_dict,
            solution_sequence=self.solution_sequence,
            i=self.i,
            whole_df_ops=self.whole_df_ops,
            seed=42  # For reproducibility
        )

    def run_probe_probs_with_restart(self):
        search_alg = ProbeATESearch(op_probs=self.op_probs)
        return run_F_experiment(
            search_algorithm=search_alg,
            df=self.df,
            common_causes=self.common_causes,
            target_ate=self.target_ate,
            epsilon=self.epsilon,
            transformations_dict=self.transformations_dict,
            solution_sequence=self.solution_sequence,
            i=self.i,
            whole_df_ops=self.whole_df_ops,
            seed=42  # For reproducibility
        )

    def run_probe_probs_with_no_restart(self):
        search_alg = ProbeATESearch(use_restart=False, op_probs=self.op_probs)
        return run_F_experiment(
            search_algorithm=search_alg,
            df=self.df,
            common_causes=self.common_causes,
            target_ate=self.target_ate,
            epsilon=self.epsilon,
            transformations_dict=self.transformations_dict,
            solution_sequence=self.solution_sequence,
            i=self.i,
            whole_df_ops=self.whole_df_ops,
            seed=42  # For reproducibility
        )
