#!/bin/bash

MAX_PARALLEL=20

MOCK_JOB_ID="209655"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M")
SCRIPT_NAME=$(basename "$0" .sh)


MAIN_OUTPUT_DIR="~/${SCRIPT_NAME}_${TIMESTAMP}"
mkdir -p "$MAIN_OUTPUT_DIR"


PROJECT_DIR="~/ATE-shifting"


source ~/my_conda_env/bin/activate


cd "$PROJECT_DIR" || exit 1

echo "Fprobe run started for experiments 39-46 (15 sub-exps each, 3 runs)."
echo "All files will be saved in ONE folder: $MAIN_OUTPUT_DIR"


TASK_ID=0


EXPS=(39 40 41 42 43 44 45 46)


get_type_for_exp() {
  case $1 in
    39|41|44)    echo "fprobe" ;;
    40|42|45)    echo "fprobe_probs_with_restart" ;;
    43|46)       echo "fprobe_probs_no_restart" ;; # ����� 46 ����� ��� ���
  esac
}

for BASE_EXP in "${EXPS[@]}"
do
    TYPE=$(get_type_for_exp "$BASE_EXP")
    
    for SUB in {1..15}
    do
        EXP="EXP${BASE_EXP}.${SUB}"

        for RUN in {1..3}
        do

            export REPETITION=$RUN

            (
                echo "Starting ${EXP}, type=${TYPE}, run ${RUN}/3 (Task $TASK_ID)"
                stdbuf -oL -eL python3 main.py "$EXP" "$TYPE"
            ) > "${MAIN_OUTPUT_DIR}/exp39_46_${MOCK_JOB_ID}_${TASK_ID}.out" 2> "${MAIN_OUTPUT_DIR}/exp39_46_${MOCK_JOB_ID}_${TASK_ID}.err" &


            TASK_ID=$((TASK_ID + 1))


            while [ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]
            do
                sleep 1
            done
        done
    done
done


wait
echo "All 360 jobs completed successfully! Results are in: $MAIN_OUTPUT_DIR"
