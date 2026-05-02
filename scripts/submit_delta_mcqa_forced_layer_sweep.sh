#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
SPLIT_SEED="${SPLIT_SEED:-0}"
FORCED_LAYER_INDICES="${FORCED_LAYER_INDICES:-17,18,19,23,24}"
TOKEN_POSITION_IDS="${TOKEN_POSITION_IDS:-last_token}"
TARGET_VARS="${TARGET_VARS:-answer_pointer,answer_token}"
DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"
TIME_LIMIT="${FORCED_LAYER_TIME:-01:00:00}"

echo "[submit-delta-force] timestamp=${TIMESTAMP}"
echo "[submit-delta-force] delta_account=${DELTA_ACCOUNT}"
echo "[submit-delta-force] delta_partition=${DELTA_PARTITION}"
echo "[submit-delta-force] split_seed=${SPLIT_SEED}"
echo "[submit-delta-force] forced_layer_indices=${FORCED_LAYER_INDICES}"
echo "[submit-delta-force] token_position_ids=${TOKEN_POSITION_IDS}"
echo "[submit-delta-force] target_vars=${TARGET_VARS}"
echo "[submit-delta-force] time_limit=${TIME_LIMIT}"

RESULTS_TIMESTAMP="${TIMESTAMP}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
SPLIT_SEED="${SPLIT_SEED}" \
FORCED_LAYER_INDICES="${FORCED_LAYER_INDICES}" \
TOKEN_POSITION_IDS="${TOKEN_POSITION_IDS}" \
TARGET_VARS="${TARGET_VARS}" \
sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --export=ALL \
  --time="${TIME_LIMIT}" \
  --job-name=mcqa-force \
  scripts/delta_mcqa_forced_layer_sweep.sbatch

echo "[submit-delta-force] results root will be ${RESULTS_ROOT}/${TIMESTAMP}_mcqa_layer_sweep"
