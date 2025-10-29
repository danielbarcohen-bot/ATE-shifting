import argparse

from experiment import Experiment
from experiments import EXPERIMENTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, help="Experiment name, e.g., EXP1")
    parser.add_argument("mode", type=str, choices=["brute", "prune"], help="Which mode to run: brute or prune")
    args = parser.parse_args()

    config = EXPERIMENTS.get(args.exp_name)
    if config is None:
        raise ValueError(f"Experiment {args.exp_name} not found. Available: {list(EXPERIMENTS.keys())}")

    experiment = Experiment(
        df=config["df"],
        common_causes=config["common_causes"],
        target_ate=config["target_ate"],
        epsilon=config["epsilon"],
        max_length=config["max_length"]
    )

    # Call the correct run method
    if args.mode == "brute":
        experiment.run_brute()
    elif args.mode == "prune":
        experiment.run_prune()

