#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
SPLIT_SEED="${SPLIT_SEED:-0}"
TOKEN_POSITION_IDS="${TOKEN_POSITION_IDS:-last_token}"
TARGET_VARS="${TARGET_VARS:-answer_pointer,answer_token}"
OT_SOURCE_TARGET_VARS="${OT_SOURCE_TARGET_VARS:-${TARGET_VARS}}"
SIGNATURE_MODE="${SIGNATURE_MODE:-family_label_delta_norm}"
OT_EPSILONS="${OT_EPSILONS:-0.5,1,2,4}"
UOT_BETA_NEURALS="${UOT_BETA_NEURALS:-0.03,0.1,0.3,1,3}"
DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"
TIME_LIMIT="${UOT_STAGE_A_TIME:-01:00:00}"

echo "[submit-delta-uot-a] timestamp=${TIMESTAMP}"
echo "[submit-delta-uot-a] delta_account=${DELTA_ACCOUNT}"
echo "[submit-delta-uot-a] delta_partition=${DELTA_PARTITION}"
echo "[submit-delta-uot-a] split_seed=${SPLIT_SEED}"
echo "[submit-delta-uot-a] token_position_ids=${TOKEN_POSITION_IDS}"
echo "[submit-delta-uot-a] target_vars=${TARGET_VARS}"
echo "[submit-delta-uot-a] ot_source_target_vars=${OT_SOURCE_TARGET_VARS}"
echo "[submit-delta-uot-a] signature_mode=${SIGNATURE_MODE}"
echo "[submit-delta-uot-a] ot_epsilons=${OT_EPSILONS}"
echo "[submit-delta-uot-a] uot_beta_neurals=${UOT_BETA_NEURALS}"
echo "[submit-delta-uot-a] time_limit=${TIME_LIMIT}"

RESULTS_TIMESTAMP="${TIMESTAMP}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
SPLIT_SEED="${SPLIT_SEED}" \
TOKEN_POSITION_IDS="${TOKEN_POSITION_IDS}" \
TARGET_VARS="${TARGET_VARS}" \
OT_SOURCE_TARGET_VARS="${OT_SOURCE_TARGET_VARS}" \
SIGNATURE_MODE="${SIGNATURE_MODE}" \
OT_EPSILONS="${OT_EPSILONS}" \
UOT_BETA_NEURALS="${UOT_BETA_NEURALS}" \
sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --export=ALL \
  --time="${TIME_LIMIT}" \
  --job-name=mcqa-uot-a \
  scripts/delta_mcqa_uot_stage_a.sbatch

echo "[submit-delta-uot-a] results root will be ${RESULTS_ROOT}/${TIMESTAMP}_mcqa"
