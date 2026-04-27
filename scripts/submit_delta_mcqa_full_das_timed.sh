#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
SPLIT_SEED="${SPLIT_SEED:-0}"

DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"

echo "[submit-delta-full-das] timestamp=${TIMESTAMP}"
echo "[submit-delta-full-das] delta_account=${DELTA_ACCOUNT}"
echo "[submit-delta-full-das] delta_partition=${DELTA_PARTITION}"
echo "[submit-delta-full-das] layers=${LAYERS:-auto}"
echo "[submit-delta-full-das] token_position_ids=${TOKEN_POSITION_IDS:-last_token}"
echo "[submit-delta-full-das] das_subspace_dims=${DAS_SUBSPACE_DIMS:-32,64,96,128,256,512,768,1024,1536,2048,2304}"
echo "[submit-delta-full-das] das_max_epochs=${DAS_MAX_EPOCHS:-100}"
echo "[submit-delta-full-das] split_seed=${SPLIT_SEED}"

RESULTS_TIMESTAMP="${TIMESTAMP}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --export=ALL \
  --job-name=mcqa-full-das \
  scripts/delta_mcqa_full_das_timed.sbatch

echo "[submit-delta-full-das] results root will be ${RESULTS_ROOT}/${TIMESTAMP}_full_das_timed_mcqa"
