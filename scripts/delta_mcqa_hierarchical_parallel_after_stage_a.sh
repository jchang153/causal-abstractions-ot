#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:?Usage: bash scripts/delta_mcqa_hierarchical_parallel_after_stage_a.sh <timestamp>}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"
ARRAY_THROTTLE_STAGE_B="${ARRAY_THROTTLE_STAGE_B:-${ARRAY_THROTTLE:-8}}"
SPLIT_SEED="${SPLIT_SEED:-0}"
SUBMIT_STAGE_C="${SUBMIT_STAGE_C:-1}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -r "${HOME}/.secrets/hf_token" ]]; then
    export HF_TOKEN="$(< "${HOME}/.secrets/hf_token")"
  else
    echo "[orchestrate-after-a] HF_TOKEN is unset and ${HOME}/.secrets/hf_token is not readable"
    exit 1
  fi
fi
export HF_HOME="${HF_HOME:-${SCRATCH:-/tmp/${USER}}/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

ROOT="${RESULTS_ROOT}/${TIMESTAMP}_mcqa_hierarchical_sweep"
mkdir -p "${ROOT}"

echo "[orchestrate-after-a] submitting Stage B arrays for timestamp=${TIMESTAMP} split_seed=${SPLIT_SEED}"
mapfile -t STAGE_B_JOBS < <(
  STAGE_A_METHOD="${STAGE_A_METHOD:-uot}" STAGE_A_HPARAM_SELECTION="${STAGE_A_HPARAM_SELECTION:-rowwise}" STAGE_B_METHODS="${STAGE_B_METHODS:-ot}" STAGE_B_SELECTION_METHODS="${STAGE_B_SELECTION_METHODS:-custom}" STAGE_B_LAYER_INDICES="${STAGE_B_LAYER_INDICES:-}" \
  UOT_BETA_NEURALS="${UOT_BETA_NEURALS:-0.03,0.1,0.3,1,3}" \
  OT_EPSILONS="${OT_EPSILONS:-0.5,1,2,4}" OT_TOP_K_VALUES="${OT_TOP_K_VALUES:-1,2,4}" \
  OT_LAMBDAS="${OT_LAMBDAS:-0.5,1,2,4}" SPLIT_SEED="${SPLIT_SEED}" ARRAY_THROTTLE="${ARRAY_THROTTLE_STAGE_B}" \
    bash scripts/submit_delta_mcqa_hierarchical_parallel_stage_b.sh "${TIMESTAMP}" 2>&1 \
    | tee "${ROOT}/stage_b_submit.log" \
    | awk '/Submitted batch job/ {print $4}'
)

if [[ "${#STAGE_B_JOBS[@]}" -eq 0 ]]; then
  echo "[orchestrate-after-a] no Stage B job ids captured"
  exit 1
fi

STAGE_B_DEPENDENCY="afterok:$(IFS=:; echo "${STAGE_B_JOBS[*]}")"
echo "[orchestrate-after-a] stage_b_jobs=${STAGE_B_JOBS[*]}"
echo "[orchestrate-after-a] submit_stage_c=${SUBMIT_STAGE_C}"
if [[ "${SUBMIT_STAGE_C}" != "1" ]]; then
  echo "[orchestrate-after-a] skipping Stage C submitter; submit aggregation after Stage B dependency=${STAGE_B_DEPENDENCY}"
  sbatch \
    --account="${DELTA_ACCOUNT}" \
    --partition="${DELTA_PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${ORCHESTRATOR_CPUS_PER_TASK:-1}" \
    --gpus-per-node="${ORCHESTRATOR_GPUS_PER_NODE:-1}" \
    --mem="${ORCHESTRATOR_MEM:-8g}" \
    --constraint="${DELTA_CONSTRAINT:-scratch}" \
    --dependency="${STAGE_B_DEPENDENCY}" \
    --time="${ORCHESTRATOR_TIME:-00:30:00}" \
    --export=ALL \
    --output="${ROOT}/mcqa-agg-%j.out" \
    --error="${ROOT}/mcqa-agg-%j.err" \
    --job-name=mcqa-agg \
    --wrap="cd ${PWD} && ALLOW_PARTIAL=1 bash scripts/aggregate_delta_mcqa_hierarchical_parallel.sh ${TIMESTAMP}"
  exit 0
fi

echo "[orchestrate-after-a] submitting Stage C submitter dependency=${STAGE_B_DEPENDENCY}"

sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${ORCHESTRATOR_CPUS_PER_TASK:-1}" \
  --gpus-per-node="${ORCHESTRATOR_GPUS_PER_NODE:-1}" \
  --mem="${ORCHESTRATOR_MEM:-8g}" \
  --constraint="${DELTA_CONSTRAINT:-scratch}" \
  --dependency="${STAGE_B_DEPENDENCY}" \
  --time="${ORCHESTRATOR_TIME:-00:30:00}" \
  --export=ALL \
  --output="${ROOT}/mcqa-submit-c-%j.out" \
  --error="${ROOT}/mcqa-submit-c-%j.err" \
  --job-name=mcqa-submit-c \
  --wrap="cd ${PWD} && STAGE_A_METHOD=${STAGE_A_METHOD:-uot} STAGE_A_HPARAM_SELECTION=${STAGE_A_HPARAM_SELECTION:-rowwise} STAGE_B_METHODS=${STAGE_B_METHODS:-ot} STAGE_B_SELECTION_METHODS=${STAGE_B_SELECTION_METHODS:-custom} STAGE_B_LAYER_INDICES=${STAGE_B_LAYER_INDICES:-} UOT_BETA_NEURALS=${UOT_BETA_NEURALS:-0.03,0.1,0.3,1,3} OT_EPSILONS=${OT_EPSILONS:-0.5,1,2,4} OT_TOP_K_VALUES=${OT_TOP_K_VALUES:-1,2,4} OT_LAMBDAS=${OT_LAMBDAS:-0.5,1,2,4} STAGE_C_TOP_CONFIGS_PER_VAR=${STAGE_C_TOP_CONFIGS_PER_VAR:-1} GUIDED_MASK_NAMES=${GUIDED_MASK_NAMES:-Selected} STAGE_C_DIM_HINT_SCALE_FACTORS=${STAGE_C_DIM_HINT_SCALE_FACTORS:-0.5,0.75,1,1.25,1.5,2} ENABLE_PCA_SUPPORT_GUIDED_DAS=${ENABLE_PCA_SUPPORT_GUIDED_DAS:-0} DISABLE_DIM_HINT_DAS=${DISABLE_DIM_HINT_DAS:-0} SPLIT_SEED=${SPLIT_SEED} ARRAY_THROTTLE_STAGE_C=${ARRAY_THROTTLE_STAGE_C:-${ARRAY_THROTTLE:-4}} bash scripts/delta_mcqa_hierarchical_parallel_after_stage_b.sh ${TIMESTAMP}"
