#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"
ARRAY_THROTTLE_STAGE_B="${ARRAY_THROTTLE_STAGE_B:-8}"
ARRAY_THROTTLE_STAGE_C="${ARRAY_THROTTLE_STAGE_C:-4}"
SUBMIT_FULL_DAS="${SUBMIT_FULL_DAS:-0}"
SUBMIT_STAGE_C="${SUBMIT_STAGE_C:-1}"
SPLIT_SEED="${SPLIT_SEED:-0}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -r "${HOME}/.secrets/hf_token" ]]; then
    export HF_TOKEN="$(< "${HOME}/.secrets/hf_token")"
  else
    echo "[submit-delta-hpar-all] HF_TOKEN is unset and ${HOME}/.secrets/hf_token is not readable"
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

echo "[submit-delta-hpar-all] timestamp=${TIMESTAMP}"
echo "[submit-delta-hpar-all] delta_account=${DELTA_ACCOUNT}"
echo "[submit-delta-hpar-all] delta_partition=${DELTA_PARTITION}"
echo "[submit-delta-hpar-all] stage_b_array_throttle=${ARRAY_THROTTLE_STAGE_B}"
echo "[submit-delta-hpar-all] stage_c_array_throttle=${ARRAY_THROTTLE_STAGE_C}"
echo "[submit-delta-hpar-all] stage_a_method=${STAGE_A_METHOD:-uot}"
echo "[submit-delta-hpar-all] stage_a_hparam_selection=${STAGE_A_HPARAM_SELECTION:-rowwise}"
echo "[submit-delta-hpar-all] stage_b_methods=${STAGE_B_METHODS:-ot}"
echo "[submit-delta-hpar-all] stage_b_selection_methods=${STAGE_B_SELECTION_METHODS:-custom}"
echo "[submit-delta-hpar-all] stage_b_layer_indices=${STAGE_B_LAYER_INDICES:-}"
echo "[submit-delta-hpar-all] uot_beta_neurals=${UOT_BETA_NEURALS:-0.03,0.1,0.3,1,3}"
echo "[submit-delta-hpar-all] hf_home=${HF_HOME}"
echo "[submit-delta-hpar-all] submit_full_das=${SUBMIT_FULL_DAS}"
echo "[submit-delta-hpar-all] submit_stage_c=${SUBMIT_STAGE_C}"
echo "[submit-delta-hpar-all] split_seed=${SPLIT_SEED}"

mapfile -t STAGE_A_JOBS < <(
  SPLIT_SEED="${SPLIT_SEED}" bash scripts/submit_delta_mcqa_hierarchical_parallel_stage_a.sh "${TIMESTAMP}" 2>&1 \
    | tee "${ROOT}/stage_a_submit.log" \
    | awk '/Submitted batch job/ {print $4}'
)

if [[ "${#STAGE_A_JOBS[@]}" -ne 1 ]]; then
  echo "[submit-delta-hpar-all] expected one Stage A job id, got: ${STAGE_A_JOBS[*]:-none}"
  exit 1
fi

STAGE_A_JOB="${STAGE_A_JOBS[0]}"
echo "[submit-delta-hpar-all] stage_a_job=${STAGE_A_JOB}"

STAGE_B_SUBMITTER_JOB="$(
  sbatch \
    --account="${DELTA_ACCOUNT}" \
    --partition="${DELTA_PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${ORCHESTRATOR_CPUS_PER_TASK:-1}" \
    --gpus-per-node="${ORCHESTRATOR_GPUS_PER_NODE:-1}" \
    --mem="${ORCHESTRATOR_MEM:-8g}" \
    --constraint="${DELTA_CONSTRAINT:-scratch}" \
    --dependency="afterok:${STAGE_A_JOB}" \
    --time="${ORCHESTRATOR_TIME:-00:30:00}" \
    --export=ALL \
    --output="${ROOT}/mcqa-submit-b-%j.out" \
    --error="${ROOT}/mcqa-submit-b-%j.err" \
    --job-name=mcqa-submit-b \
    --wrap="cd ${PWD} && STAGE_A_METHOD=${STAGE_A_METHOD:-uot} STAGE_A_HPARAM_SELECTION=${STAGE_A_HPARAM_SELECTION:-rowwise} STAGE_B_METHODS=${STAGE_B_METHODS:-ot} STAGE_B_SELECTION_METHODS=${STAGE_B_SELECTION_METHODS:-custom} STAGE_B_LAYER_INDICES=${STAGE_B_LAYER_INDICES:-} UOT_BETA_NEURALS=${UOT_BETA_NEURALS:-0.03,0.1,0.3,1,3} OT_EPSILONS=${OT_EPSILONS:-0.5,1,2,4} OT_TOP_K_VALUES=${OT_TOP_K_VALUES:-1,2,4} OT_LAMBDAS=${OT_LAMBDAS:-0.5,1,2,4} SPLIT_SEED=${SPLIT_SEED} SUBMIT_STAGE_C=${SUBMIT_STAGE_C} ARRAY_THROTTLE_STAGE_B=${ARRAY_THROTTLE_STAGE_B} ARRAY_THROTTLE_STAGE_C=${ARRAY_THROTTLE_STAGE_C} bash scripts/delta_mcqa_hierarchical_parallel_after_stage_a.sh ${TIMESTAMP}" \
    | awk '/Submitted batch job/ {print $4}'
)"
echo "[submit-delta-hpar-all] stage_b_submitter_job=${STAGE_B_SUBMITTER_JOB}"

if [[ "${SUBMIT_FULL_DAS}" == "1" ]]; then
  FULL_DAS_TIMESTAMP="${FULL_DAS_TIMESTAMP:-${TIMESTAMP}_full_das}"
  mapfile -t FULL_DAS_JOBS < <(
    SPLIT_SEED="${SPLIT_SEED}" bash scripts/submit_delta_mcqa_full_das_timed.sh "${FULL_DAS_TIMESTAMP}" 2>&1 \
      | tee "${ROOT}/full_das_submit.log" \
      | awk '/Submitted batch job/ {print $4}'
  )
  echo "[submit-delta-hpar-all] full_das_jobs=${FULL_DAS_JOBS[*]:-none}"
fi

echo "[submit-delta-hpar-all] chain submitted"
echo "[submit-delta-hpar-all] results root: ${ROOT}"
