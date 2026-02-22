import argparse

from experiment import Experiment, RandomExperiment
from experiments import EXPERIMENTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, help="Experiment name, e.g., EXP1")
    parser.add_argument("mode", type=str, choices=["brute", "prune", "parallel", "astar", "probe", "prune_no_bit_mask", "probe_lin_reg", "llm_zero_shot", "llm_few_shot", "llm_zero_shot_cot", "llm_few_shot_cot", "random", "probe_no_hash", "prune_no_hash", "probe_no_bit_mask", "probe_no_lazy_eval"],
                        help="Which mode to run: brute prune or parallel prune")
    args = parser.parse_args()
    config = EXPERIMENTS.get(args.exp_name)
    if config is None:
        raise ValueError(f"Experiment {args.exp_name} not found. Available: {list(EXPERIMENTS.keys())}")
    if args.mode == "random":
        experiment = RandomExperiment(
            df=config["df"],
            transformations_dict=config["transformations_dict"],
            common_causes=config["common_causes"],
            target_ate=config["target_ate"],
            epsilon=config["epsilon"],
            sequence_length=config["sequence_length"]
        )
        experiment.run_random()
    else:
        experiment = Experiment(
            df=config["df"],
            transformations_dict=config["transformations_dict"],
            common_causes=config["common_causes"],
            target_ate=config["target_ate"],
            epsilon=config["epsilon"],
            max_length=config["max_length"]
        )
        # Call the correct run method
        if args.mode == "brute":
            experiment.run_brute()
        if args.mode == "prune":
            experiment.run_prune()
        # if args.mode == "parallel":
        #     experiment.run_parallel_prune()
        # if args.mode == "astar":
        #     experiment.run_AStar()
        if args.mode == "probe":
            experiment.run_probe()
        if args.mode == "probe_no_hash":
            experiment.run_probe_no_hash()
        if args.mode == "prune_no_hash":
            experiment.run_prune_no_hash()
        if args.mode == "probe_no_bit_mask":
            experiment.run_probe_no_bit_mask()
        if args.mode == "prune_no_bit_mask":
            experiment.run_prune_no_bit_mask()
        if args.mode == "probe_no_lazy_eval":
            experiment.run_probe_no_lazy_eval()
        if args.mode == "llm_zero_shot":
            experiment.run_llm_zero_shot()
        if args.mode == "llm_few_shot":
            experiment.run_llm_few_shot()
        if args.mode == "llm_zero_shot_cot":
            experiment.run_llm_zero_shot(True)
        if args.mode == "llm_few_shot_cot":
            experiment.run_llm_few_shot(True)
        elif args.mode == "probe_lin_reg":
            experiment.run_probe_line_reg_heuristic()
