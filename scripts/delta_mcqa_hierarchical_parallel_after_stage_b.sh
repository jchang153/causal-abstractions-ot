#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:?Usage: bash scripts/delta_mcqa_hierarchical_parallel_after_stage_b.sh <timestamp>}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"
ARRAY_THROTTLE_STAGE_C="${ARRAY_THROTTLE_STAGE_C:-${ARRAY_THROTTLE:-4}}"
SPLIT_SEED="${SPLIT_SEED:-0}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -r "${HOME}/.secrets/hf_token" ]]; then
    export HF_TOKEN="$(< "${HOME}/.secrets/hf_token")"
  else
    echo "[orchestrate-after-b] HF_TOKEN is unset and ${HOME}/.secrets/hf_token is not readable"
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

echo "[orchestrate-after-b] submitting Stage C arrays for timestamp=${TIMESTAMP} split_seed=${SPLIT_SEED}"
mapfile -t STAGE_C_JOBS < <(
  STAGE_A_METHOD="${STAGE_A_METHOD:-uot}" STAGE_B_METHODS="${STAGE_B_METHODS:-ot}" STAGE_B_SELECTION_METHODS="${STAGE_B_SELECTION_METHODS:-custom}" STAGE_B_LAYER_INDICES="${STAGE_B_LAYER_INDICES:-}" \
  UOT_BETA_NEURALS="${UOT_BETA_NEURALS:-0.03,0.1,0.3,1,3}" \
  OT_EPSILONS="${OT_EPSILONS:-0.5,1,2,4}" OT_TOP_K_VALUES="${OT_TOP_K_VALUES:-1,2,4}" \
  OT_LAMBDAS="${OT_LAMBDAS:-0.5,1,2,4}" SPLIT_SEED="${SPLIT_SEED}" ARRAY_THROTTLE="${ARRAY_THROTTLE_STAGE_C}" \
    bash scripts/submit_delta_mcqa_hierarchical_parallel_stage_c.sh "${TIMESTAMP}" 2>&1 \
    | tee "${ROOT}/stage_c_submit.log" \
    | awk '/Submitted batch job/ {print $4}'
)

if [[ "${#STAGE_C_JOBS[@]}" -eq 0 ]]; then
  echo "[orchestrate-after-b] no Stage C job ids captured; submitting aggregation without Stage C dependency"
  AGG_DEPENDENCY_ARGS=()
else
  STAGE_C_DEPENDENCY="afterok:$(IFS=:; echo "${STAGE_C_JOBS[*]}")"
  echo "[orchestrate-after-b] stage_c_jobs=${STAGE_C_JOBS[*]}"
  echo "[orchestrate-after-b] submitting aggregation dependency=${STAGE_C_DEPENDENCY}"
  AGG_DEPENDENCY_ARGS=(--dependency="${STAGE_C_DEPENDENCY}")
fi

sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${ORCHESTRATOR_CPUS_PER_TASK:-1}" \
  --gpus-per-node="${ORCHESTRATOR_GPUS_PER_NODE:-1}" \
  --mem="${ORCHESTRATOR_MEM:-8g}" \
  --constraint="${DELTA_CONSTRAINT:-scratch}" \
  "${AGG_DEPENDENCY_ARGS[@]}" \
  --time="${ORCHESTRATOR_TIME:-00:30:00}" \
  --export=ALL \
  --output="${ROOT}/mcqa-agg-%j.out" \
  --error="${ROOT}/mcqa-agg-%j.err" \
  --job-name=mcqa-agg \
  --wrap="cd ${PWD} && STAGE_A_METHOD=${STAGE_A_METHOD:-uot} STAGE_A_HPARAM_SELECTION=${STAGE_A_HPARAM_SELECTION:-rowwise} STAGE_B_METHODS=${STAGE_B_METHODS:-ot} STAGE_B_SELECTION_METHODS=${STAGE_B_SELECTION_METHODS:-custom} STAGE_B_LAYER_INDICES=${STAGE_B_LAYER_INDICES:-} UOT_BETA_NEURALS=${UOT_BETA_NEURALS:-0.03,0.1,0.3,1,3} OT_EPSILONS=${OT_EPSILONS:-0.5,1,2,4} OT_TOP_K_VALUES=${OT_TOP_K_VALUES:-1,2,4} OT_LAMBDAS=${OT_LAMBDAS:-0.5,1,2,4} bash scripts/aggregate_delta_mcqa_hierarchical_parallel.sh ${TIMESTAMP}"
