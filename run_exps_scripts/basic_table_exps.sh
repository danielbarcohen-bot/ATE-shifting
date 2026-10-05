#!/bin/bash

MAX_PARALLEL=20

MOCK_JOB_ID="209652"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M")

SCRIPT_NAME=$(basename "$0" .sh)

OUTPUT_DIR="~/${SCRIPT_NAME}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

PROJECT_DIR="~/ATE-shifting"

source ~/my_conda_env/bin/activate

EXPERIMENTS=(
  27
  27_lal_zero
  28
  35
)

TYPES=(
  "brute"
  "probe_brute"
  "probe"
  "probe_probs_no_restart"
  "probe_probs_with_restart"
)

NUM_EXPS=${#EXPERIMENTS[@]}
NUM_TYPES=${#TYPES[@]}
RUNS_PER_EXP=3
CONFIGS_PER_EXP=$((NUM_TYPES * RUNS_PER_EXP))

echo "New run started!"
echo "Output files will be saved in a unique folder: $OUTPUT_DIR"

cd "$PROJECT_DIR" || exit 1

for i in {0..59}
do
    EXP_INDEX=$((i / CONFIGS_PER_EXP))
    CONFIG_INDEX=$((i % CONFIGS_PER_EXP))

    TYPE_INDEX=$((CONFIG_INDEX / RUNS_PER_EXP))
    RUN_INDEX=$((CONFIG_INDEX % RUNS_PER_EXP))

    EXP="EXP${EXPERIMENTS[$EXP_INDEX]}"
    TYPE="${TYPES[$TYPE_INDEX]}"
    RUN=$((RUN_INDEX + 1))


    (
        echo "Starting ${EXP}, type=${TYPE}, run ${RUN}/3 (Task $i)"
        stdbuf -oL -eL python3 main.py "$EXP" "$TYPE"
    ) > "${OUTPUT_DIR}/exp_larger_${MOCK_JOB_ID}_${i}.out" 2> "${OUTPUT_DIR}/exp_larger_${MOCK_JOB_ID}_${i}.err" &

    while [ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]
    do
        sleep 2
    done
done

wait
echo "All jobs completed successfully! Results are in: $OUTPUT_DIR"
