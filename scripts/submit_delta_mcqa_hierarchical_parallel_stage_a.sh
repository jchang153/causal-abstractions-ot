#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
SPLIT_SEED="${SPLIT_SEED:-0}"

DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"

echo "[submit-delta-hpar-a] timestamp=${TIMESTAMP}"
echo "[submit-delta-hpar-a] delta_account=${DELTA_ACCOUNT}"
echo "[submit-delta-hpar-a] delta_partition=${DELTA_PARTITION}"
echo "[submit-delta-hpar-a] stage_a_token_position_ids=${STAGE_A_TOKEN_POSITION_IDS:-last_token}"
echo "[submit-delta-hpar-a] stage_a_method=${STAGE_A_METHOD:-ot}"
echo "[submit-delta-hpar-a] uot_beta_neurals=${UOT_BETA_NEURALS:-0.03,0.1,0.3,1,3}"
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
