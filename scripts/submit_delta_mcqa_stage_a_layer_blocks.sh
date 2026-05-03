#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
SPLIT_SEED="${SPLIT_SEED:-0}"
LAYER_BLOCKS="${LAYER_BLOCKS:-0-4;5-9;10-14;15-19;20-25}"
TOKEN_POSITION_IDS="${TOKEN_POSITION_IDS:-last_token}"
SIGNATURE_MODE="${SIGNATURE_MODE:-family_label_delta_norm}"
TARGET_VARS="${TARGET_VARS:-answer_pointer,answer_token}"
OT_SOURCE_TARGET_VARS="${OT_SOURCE_TARGET_VARS:-${TARGET_VARS}}"
DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"
TIME_LIMIT="${STAGE_A_BLOCK_TIME:-01:00:00}"

echo "[submit-delta-block-a] timestamp=${TIMESTAMP}"
echo "[submit-delta-block-a] delta_account=${DELTA_ACCOUNT}"
echo "[submit-delta-block-a] delta_partition=${DELTA_PARTITION}"
echo "[submit-delta-block-a] split_seed=${SPLIT_SEED}"
echo "[submit-delta-block-a] layer_blocks=${LAYER_BLOCKS}"
echo "[submit-delta-block-a] token_position_ids=${TOKEN_POSITION_IDS}"
echo "[submit-delta-block-a] signature_mode=${SIGNATURE_MODE}"
echo "[submit-delta-block-a] target_vars=${TARGET_VARS}"
echo "[submit-delta-block-a] ot_source_target_vars=${OT_SOURCE_TARGET_VARS}"
echo "[submit-delta-block-a] time_limit=${TIME_LIMIT}"

RESULTS_TIMESTAMP="${TIMESTAMP}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
SPLIT_SEED="${SPLIT_SEED}" \
LAYER_BLOCKS="${LAYER_BLOCKS}" \
TOKEN_POSITION_IDS="${TOKEN_POSITION_IDS}" \
SIGNATURE_MODE="${SIGNATURE_MODE}" \
TARGET_VARS="${TARGET_VARS}" \
OT_SOURCE_TARGET_VARS="${OT_SOURCE_TARGET_VARS}" \
sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --export=ALL \
  --time="${TIME_LIMIT}" \
  --job-name=mcqa-block-a \
  scripts/delta_mcqa_stage_a_layer_blocks.sbatch

echo "[submit-delta-block-a] results root will be ${RESULTS_ROOT}/${TIMESTAMP}_mcqa"
