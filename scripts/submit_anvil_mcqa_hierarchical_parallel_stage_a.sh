#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/anvil}"
SPLIT_SEED="${SPLIT_SEED:-0}"

ANVIL_ACCOUNT="${ANVIL_ACCOUNT:-cis260602-ai}"
ANVIL_PARTITION="${ANVIL_PARTITION:-ai}"
STAGE_A_TIME="${STAGE_A_TIME:-01:00:00}"

echo "[submit-anvil-hpar-a] timestamp=${TIMESTAMP}"
echo "[submit-anvil-hpar-a] ANVIL_ACCOUNT=${ANVIL_ACCOUNT}"
echo "[submit-anvil-hpar-a] ANVIL_PARTITION=${ANVIL_PARTITION}"
echo "[submit-anvil-hpar-a] stage_a_time=${STAGE_A_TIME}"
echo "[submit-anvil-hpar-a] stage_a_token_position_ids=${STAGE_A_TOKEN_POSITION_IDS:-last_token}"
echo "[submit-anvil-hpar-a] split_seed=${SPLIT_SEED}"

RESULTS_TIMESTAMP="${TIMESTAMP}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
SPLIT_SEED="${SPLIT_SEED}" \
sbatch \
  --account="${ANVIL_ACCOUNT}" \
  --partition="${ANVIL_PARTITION}" \
  --time="${STAGE_A_TIME}" \
  --export=ALL \
  --job-name=mcqa-hpar-a \
  scripts/anvil_mcqa_hierarchical_parallel_stage_a.sbatch

echo "[submit-anvil-hpar-a] results root will be ${RESULTS_ROOT}/${TIMESTAMP}_mcqa_hierarchical_sweep"
