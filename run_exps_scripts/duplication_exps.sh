#!/bin/bash

MAX_PARALLEL=15

MOCK_JOB_ID="209653"


TIMESTAMP=$(date +"%Y-%m-%d_%H-%M")


SCRIPT_NAME=$(basename "$0" .sh)


OUTPUT_DIR="~/${SCRIPT_NAME}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"


PROJECT_DIR="~/ATE-shifting"


source ~/my_conda_env/bin/activate


EXPERIMENTS=(
  33.1 33.2 33.3 33.4 33.5
  34.1 34.2 34.3 34.4 34.5
  38.1 38.2 38.3 38.4 38.5
)

TYPES=("probe" "probe_probs_no_restart" "probe_probs_with_restart")

NUM_EXPS=${#EXPERIMENTS[@]}
NUM_TYPES=${#TYPES[@]}
RUNS_PER_EXP=3
CONFIGS_PER_EXP=$((NUM_TYPES * RUNS_PER_EXP))


TOTAL_JOBS=$((NUM_EXPS * CONFIGS_PER_EXP))
MAX_LOOP_INDEX=$((TOTAL_JOBS - 1))

echo "Scale run started for $NUM_EXPS experiments! Total jobs: $TOTAL_JOBS"
echo "Output files will be saved in a unique folder: $OUTPUT_DIR"


cd "$PROJECT_DIR" || exit 1


for i in $(eval echo "{0..$MAX_LOOP_INDEX}")
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
    ) > "${OUTPUT_DIR}/exp_scale_${MOCK_JOB_ID}_${i}.out" 2> "${OUTPUT_DIR}/exp_scale_${MOCK_JOB_ID}_${i}.err" &


    while [ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]
    do
        sleep 2
    done
done


wait
echo "All $TOTAL_JOBS scale jobs completed successfully! Results are in: $OUTPUT_DIR"
