#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/runpod_hpar_local_exec.sh

STAGE="${1:?usage: $0 stage-a|stage-b|stage-c}"
RUN_TAG="${RUN_TAG:-20260506_mcqa_runpod_final_confirm}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR"

layers_for_seed() {
  case "$1" in
    0) echo "${SEED0_LAYERS:-17,24}" ;;
    1) echo "${SEED1_LAYERS:-17,24}" ;;
    2) echo "${SEED2_LAYERS:-18,23}" ;;
    *) echo "unknown seed $1" >&2; exit 1 ;;
  esac
}

common_args() {
  local seed="$1"
  local root="$2"
  local ts="$3"

  echo \
    --device cuda \
    --dataset-path jchang153/copycolors_mcqa \
    --batch-size 64 \
    --results-root "$root" \
    --results-timestamp "$ts" \
    --signatures-dir signatures \
    --stage-a-token-position-ids last_token \
    --target-vars answer_pointer,answer_token \
    --stage-a-method ot,uot \
    --stage-a-hparam-selection rowwise \
    --stage-a-rerank-top-k 6 \
    --stage-a-rerank-drop-ratio 0 \
    --stage-a-rerank-min-k 6 \
    --stage-a-rerank-max-k 8 \
    --signature-mode family_label_delta_norm \
    --ot-epsilons 1 \
    --uot-beta-neurals 0.03,0.1,0.3,1,3 \
    --ot-top-k-values 1,2,4 \
    --ot-lambdas 0.5,1,2,4 \
    --calibration-metric family_weighted_macro_exact_acc \
    --calibration-family-weights 1,1.5,2 \
    --split-seed "$seed"
}

for SEED in 0 1 2; do
  LAYERS="$(layers_for_seed "$SEED")"
  TS="${RUN_TAG}_seed${SEED}_plot5_stageA_layers"
  ROOT="/workspace/results/runpod_final_confirm_seed${SEED}_plot5_stageA_layers"
  SWEEP="${ROOT}/${TS}_mcqa_hierarchical_sweep"
  LOG="${LOG_DIR}/${TS}_${STAGE}.log"

  echo "===== RunPod ${STAGE} seed=${SEED} layers=${LAYERS} =====" | tee "$LOG"
  START="$(date +%s)"

  if [[ "$STAGE" == "stage-a" ]]; then
    python -u mcqa_delta_hierarchical_parallel.py stage-a $(common_args "$SEED" "$ROOT" "$TS") 2>&1 | tee -a "$LOG"
  elif [[ "$STAGE" == "stage-b" ]]; then
    python -u mcqa_delta_hierarchical_parallel.py plan-stage-b \
      $(common_args "$SEED" "$ROOT" "$TS") \
      --stage-b-methods ot \
      --stage-b-selection-methods fixed \
      --stage-b-layer-indices "$LAYERS" \
      2>&1 | tee -a "$LOG"

    run_task_file "$SWEEP/stage_b_native_tasks.json" "seed${SEED}-stageB-native" 2>&1 | tee -a "$LOG"
    run_task_file "$SWEEP/stage_b_pca_tasks.json" "seed${SEED}-stageB-pca" 2>&1 | tee -a "$LOG"
  elif [[ "$STAGE" == "stage-c" ]]; then
    python -u mcqa_delta_hierarchical_parallel.py aggregate-stage-b $(common_args "$SEED" "$ROOT" "$TS") 2>&1 | tee -a "$LOG"

    python -u mcqa_delta_hierarchical_parallel.py plan-stage-c \
      $(common_args "$SEED" "$ROOT" "$TS") \
      --stage-b-layer-indices "$LAYERS" \
      --stage-c-top-configs-per-var 1 \
      2>&1 | tee -a "$LOG"

    run_task_file "$SWEEP/stage_c_a_only_tasks.json" "seed${SEED}-stageC-a-only" 2>&1 | tee -a "$LOG"
    run_task_file "$SWEEP/stage_c_native_dim_tasks.json" "seed${SEED}-stageC-native-dim" 2>&1 | tee -a "$LOG"
    run_task_file "$SWEEP/stage_c_pca_dim_tasks.json" "seed${SEED}-stageC-pca-dim" 2>&1 | tee -a "$LOG"

    python -u mcqa_delta_hierarchical_parallel.py aggregate-stage-c $(common_args "$SEED" "$ROOT" "$TS") 2>&1 | tee -a "$LOG"
  else
    echo "unknown stage: $STAGE" >&2
    exit 1
  fi

  END="$(date +%s)"
  echo "[time] ${STAGE} seed=${SEED} seconds=$((END - START))" | tee -a "$LOG"
done
