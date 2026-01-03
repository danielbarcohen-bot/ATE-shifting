import argparse

from experiment import Experiment
from experiments import EXPERIMENTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, help="Experiment name, e.g., EXP1")
    parser.add_argument("mode", type=str, choices=["brute", "prune", "parallel", "astar", "probe", "probe_lin_reg"],
                        help="Which mode to run: brute prune or parallel prune")
    args = parser.parse_args()
    config = EXPERIMENTS.get(args.exp_name)
    if config is None:
        raise ValueError(f"Experiment {args.exp_name} not found. Available: {list(EXPERIMENTS.keys())}")

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
    if args.mode == "parallel":
        experiment.run_parallel_prune()
    if args.mode == "astar":
        experiment.run_AStar()
    if args.mode == "probe":
        experiment.run_probe()
    elif args.mode == "probe_lin_reg":
        experiment.run_probe_line_reg_heuristic()
