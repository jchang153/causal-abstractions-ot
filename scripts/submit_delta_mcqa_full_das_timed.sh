#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
SPLIT_SEED="${SPLIT_SEED:-0}"
export DAS_RESTARTS="${DAS_RESTARTS:-2}"

DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -r "${HOME}/.secrets/hf_token" ]]; then
    export HF_TOKEN="$(< "${HOME}/.secrets/hf_token")"
  else
    echo "[submit-delta-full-das] HF_TOKEN is unset and ${HOME}/.secrets/hf_token is not readable"
    exit 1
  fi
fi
export HF_HOME="${HF_HOME:-${SCRATCH:-/tmp/${USER}}/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "[submit-delta-full-das] timestamp=${TIMESTAMP}"
echo "[submit-delta-full-das] delta_account=${DELTA_ACCOUNT}"
echo "[submit-delta-full-das] delta_partition=${DELTA_PARTITION}"
echo "[submit-delta-full-das] layers=${LAYERS:-auto}"
echo "[submit-delta-full-das] token_position_ids=${TOKEN_POSITION_IDS:-last_token}"
echo "[submit-delta-full-das] das_subspace_dims=${DAS_SUBSPACE_DIMS:-32,64,96,128,256,512,768,1024,1536,2048,2304}"
echo "[submit-delta-full-das] das_max_epochs=${DAS_MAX_EPOCHS:-100}"
echo "[submit-delta-full-das] das_restarts=${DAS_RESTARTS}"
echo "[submit-delta-full-das] hf_home=${HF_HOME}"
echo "[submit-delta-full-das] split_seed=${SPLIT_SEED}"

RESULTS_TIMESTAMP="${TIMESTAMP}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
SPLIT_SEED="${SPLIT_SEED}" \
sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --export=ALL \
  --job-name=mcqa-full-das \
  scripts/delta_mcqa_full_das_timed.sbatch

echo "[submit-delta-full-das] results root will be ${RESULTS_ROOT}/${TIMESTAMP}_full_das_timed_mcqa"
