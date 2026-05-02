#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/anvil}"
SPLIT_SEED="${SPLIT_SEED:-0}"
FORCED_LAYER_INDICES="${FORCED_LAYER_INDICES:-17,18,19,23,24}"
TOKEN_POSITION_IDS="${TOKEN_POSITION_IDS:-last_token}"
TARGET_VARS="${TARGET_VARS:-answer_pointer,answer_token}"
ANVIL_ACCOUNT="${ANVIL_ACCOUNT:-cis260602-ai}"
ANVIL_PARTITION="${ANVIL_PARTITION:-ai}"
TIME_LIMIT="${FORCED_LAYER_TIME:-01:00:00}"

echo "[submit-anvil-force] timestamp=${TIMESTAMP}"
echo "[submit-anvil-force] ANVIL_ACCOUNT=${ANVIL_ACCOUNT}"
echo "[submit-anvil-force] ANVIL_PARTITION=${ANVIL_PARTITION}"
echo "[submit-anvil-force] split_seed=${SPLIT_SEED}"
echo "[submit-anvil-force] forced_layer_indices=${FORCED_LAYER_INDICES}"
echo "[submit-anvil-force] token_position_ids=${TOKEN_POSITION_IDS}"
echo "[submit-anvil-force] target_vars=${TARGET_VARS}"
echo "[submit-anvil-force] time_limit=${TIME_LIMIT}"

RESULTS_TIMESTAMP="${TIMESTAMP}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
SPLIT_SEED="${SPLIT_SEED}" \
FORCED_LAYER_INDICES="${FORCED_LAYER_INDICES}" \
TOKEN_POSITION_IDS="${TOKEN_POSITION_IDS}" \
TARGET_VARS="${TARGET_VARS}" \
sbatch \
  --account="${ANVIL_ACCOUNT}" \
  --partition="${ANVIL_PARTITION}" \
  --export=ALL \
  --time="${TIME_LIMIT}" \
  --job-name=mcqa-force \
  scripts/anvil_mcqa_forced_layer_sweep.sbatch

echo "[submit-anvil-force] results root will be ${RESULTS_ROOT}/${TIMESTAMP}_mcqa_layer_sweep"
