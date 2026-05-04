#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
SPLIT_SEED="${SPLIT_SEED:-0}"

DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -r "${HOME}/.secrets/hf_token" ]]; then
    export HF_TOKEN="$(< "${HOME}/.secrets/hf_token")"
  else
    echo "[submit-delta-hpar-a] HF_TOKEN is unset and ${HOME}/.secrets/hf_token is not readable"
    exit 1
  fi
fi
export HF_HOME="${HF_HOME:-${SCRATCH:-/tmp/${USER}}/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "[submit-delta-hpar-a] timestamp=${TIMESTAMP}"
echo "[submit-delta-hpar-a] delta_account=${DELTA_ACCOUNT}"
echo "[submit-delta-hpar-a] delta_partition=${DELTA_PARTITION}"
echo "[submit-delta-hpar-a] stage_a_token_position_ids=${STAGE_A_TOKEN_POSITION_IDS:-last_token}"
echo "[submit-delta-hpar-a] stage_a_method=${STAGE_A_METHOD:-uot}"
echo "[submit-delta-hpar-a] stage_a_hparam_selection=${STAGE_A_HPARAM_SELECTION:-rowwise}"
echo "[submit-delta-hpar-a] stage_a_rerank_top_k=${STAGE_A_RERANK_TOP_K:-0}"
echo "[submit-delta-hpar-a] uot_beta_neurals=${UOT_BETA_NEURALS:-0.03,0.1,0.3,1,3}"
echo "[submit-delta-hpar-a] hf_home=${HF_HOME}"
echo "[submit-delta-hpar-a] split_seed=${SPLIT_SEED}"

RESULTS_TIMESTAMP="${TIMESTAMP}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
SPLIT_SEED="${SPLIT_SEED}" \
sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --export=ALL \
  --job-name=mcqa-hpar-a \
  scripts/delta_mcqa_hierarchical_parallel_stage_a.sbatch

echo "[submit-delta-hpar-a] results root will be ${RESULTS_ROOT}/${TIMESTAMP}_mcqa_hierarchical_sweep"
