#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/anvil}"
SPLIT_SEED="${SPLIT_SEED:-0}"
LAYER_BLOCKS="${LAYER_BLOCKS:-0-4;5-9;10-14;15-19;20-25}"
TOKEN_POSITION_IDS="${TOKEN_POSITION_IDS:-last_token}"
SIGNATURE_MODE="${SIGNATURE_MODE:-family_label_delta_norm}"
TARGET_VARS="${TARGET_VARS:-answer_pointer,answer_token}"
OT_SOURCE_TARGET_VARS="${OT_SOURCE_TARGET_VARS:-${TARGET_VARS}}"
ANVIL_ACCOUNT="${ANVIL_ACCOUNT:-cis260602-ai}"
ANVIL_PARTITION="${ANVIL_PARTITION:-ai}"
TIME_LIMIT="${STAGE_A_BLOCK_TIME:-01:00:00}"

echo "[submit-anvil-block-a] timestamp=${TIMESTAMP}"
echo "[submit-anvil-block-a] ANVIL_ACCOUNT=${ANVIL_ACCOUNT}"
echo "[submit-anvil-block-a] ANVIL_PARTITION=${ANVIL_PARTITION}"
echo "[submit-anvil-block-a] split_seed=${SPLIT_SEED}"
echo "[submit-anvil-block-a] layer_blocks=${LAYER_BLOCKS}"
echo "[submit-anvil-block-a] token_position_ids=${TOKEN_POSITION_IDS}"
echo "[submit-anvil-block-a] signature_mode=${SIGNATURE_MODE}"
echo "[submit-anvil-block-a] target_vars=${TARGET_VARS}"
echo "[submit-anvil-block-a] ot_source_target_vars=${OT_SOURCE_TARGET_VARS}"
echo "[submit-anvil-block-a] time_limit=${TIME_LIMIT}"

RESULTS_TIMESTAMP="${TIMESTAMP}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
SPLIT_SEED="${SPLIT_SEED}" \
LAYER_BLOCKS="${LAYER_BLOCKS}" \
TOKEN_POSITION_IDS="${TOKEN_POSITION_IDS}" \
SIGNATURE_MODE="${SIGNATURE_MODE}" \
TARGET_VARS="${TARGET_VARS}" \
OT_SOURCE_TARGET_VARS="${OT_SOURCE_TARGET_VARS}" \
sbatch \
  --account="${ANVIL_ACCOUNT}" \
  --partition="${ANVIL_PARTITION}" \
  --export=ALL \
  --time="${TIME_LIMIT}" \
  --job-name=mcqa-block-a \
  scripts/anvil_mcqa_stage_a_layer_blocks.sbatch

echo "[submit-anvil-block-a] results root will be ${RESULTS_ROOT}/${TIMESTAMP}_mcqa"
