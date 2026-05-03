#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
SPLIT_SEED="${SPLIT_SEED:-0}"
TOKEN_POSITION_IDS="${TOKEN_POSITION_IDS:-last_token}"
TARGET_VARS="${TARGET_VARS:-answer_pointer,answer_token}"
OT_SOURCE_TARGET_VARS="${OT_SOURCE_TARGET_VARS:-answer_pointer,answer_token,null}"
SIGNATURE_MODE="${SIGNATURE_MODE:-family_label_delta_norm}"
DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"
TIME_LIMIT="${NULL_STAGE_A_TIME:-01:00:00}"

echo "[submit-delta-null-a] timestamp=${TIMESTAMP}"
echo "[submit-delta-null-a] delta_account=${DELTA_ACCOUNT}"
echo "[submit-delta-null-a] delta_partition=${DELTA_PARTITION}"
echo "[submit-delta-null-a] split_seed=${SPLIT_SEED}"
echo "[submit-delta-null-a] token_position_ids=${TOKEN_POSITION_IDS}"
echo "[submit-delta-null-a] target_vars=${TARGET_VARS}"
echo "[submit-delta-null-a] ot_source_target_vars=${OT_SOURCE_TARGET_VARS}"
echo "[submit-delta-null-a] signature_mode=${SIGNATURE_MODE}"
echo "[submit-delta-null-a] time_limit=${TIME_LIMIT}"

RESULTS_TIMESTAMP="${TIMESTAMP}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
SPLIT_SEED="${SPLIT_SEED}" \
TOKEN_POSITION_IDS="${TOKEN_POSITION_IDS}" \
TARGET_VARS="${TARGET_VARS}" \
OT_SOURCE_TARGET_VARS="${OT_SOURCE_TARGET_VARS}" \
SIGNATURE_MODE="${SIGNATURE_MODE}" \
sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --export=ALL \
  --time="${TIME_LIMIT}" \
  --job-name=mcqa-null-a \
  scripts/delta_mcqa_null_stage_a.sbatch

echo "[submit-delta-null-a] results root will be ${RESULTS_ROOT}/${TIMESTAMP}_mcqa"
