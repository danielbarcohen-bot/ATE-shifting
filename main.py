import argparse

from experiment import Experiment, RandomExperiment, ProbabilitiesExperiment
from experiments import EXPERIMENTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, help="Experiment name, e.g., EXP1")
    parser.add_argument("mode", type=str,
                        choices=["brute", "brute_prob", "oe", "parallel", "astar", "probe", "probe_brute", "oe_no_bit_mask", "llm_zero_shot",
                                 "llm_few_shot", "llm_zero_shot_cot", "llm_few_shot_cot", "random", "probe_no_hash",
                                 "oe_no_hash", "probe_probs_no_restart", "probe_probs_with_restart", "greedy"],
                        help="Which mode to run: brute OE or probe")
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
    if args.mode in ["probe_probs_no_restart", "probe_probs_with_restart", "greedy", "brute_prob", "probe_brute"]:
        experiment = ProbabilitiesExperiment(
            df=config["df"],
            transformations_dict=config["transformations_dict"],
            common_causes=config["common_causes"],
            target_ate=config["target_ate"],
            epsilon=config["epsilon"],
            max_sequence_length=config["max_length"],
            op_probs=config["op_probs"]
        )
        if args.mode == "probe_probs_no_restart":
            experiment.run_probe_no_restart()
        if args.mode == "probe_probs_with_restart":
            experiment.run_probe()
        if args.mode == "greedy":
            experiment.run_greedy()
        if args.mode == "brute_prob":
            experiment.run_brute_prob()
        if args.mode == "probe_brute":
            experiment.run_probe_brute()
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
        if args.mode == "oe":
            experiment.run_prune()
        # if args.mode == "astar":
        #     experiment.run_AStar()
        if args.mode == "probe":
            experiment.run_probe()
        if args.mode == "probe_no_hash":
            experiment.run_probe_no_hash()
        if args.mode == "oe_no_hash":
            experiment.run_prune_no_hash()
        if args.mode == "oe_no_bit_mask":
            experiment.run_prune_no_bit_mask()
        if args.mode == "llm_zero_shot":
            experiment.run_llm_zero_shot()
        if args.mode == "llm_few_shot":
            experiment.run_llm_few_shot()
        if args.mode == "llm_zero_shot_cot":
            experiment.run_llm_zero_shot(True)
        elif args.mode == "llm_few_shot_cot":
            experiment.run_llm_few_shot(True)
