#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/anvil}"
SPLIT_SEED="${SPLIT_SEED:-0}"

ANVIL_ACCOUNT="${ANVIL_ACCOUNT:-cis260602-ai}"
ANVIL_PARTITION="${ANVIL_PARTITION:-ai}"
STAGE_A_TIME="${STAGE_A_TIME:-01:00:00}"
STAGE_A_JOB_NAME="${STAGE_A_JOB_NAME:-mcqa-hpar-a}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"

echo "[submit-anvil-hpar-a] timestamp=${TIMESTAMP}"
echo "[submit-anvil-hpar-a] ANVIL_ACCOUNT=${ANVIL_ACCOUNT}"
echo "[submit-anvil-hpar-a] ANVIL_PARTITION=${ANVIL_PARTITION}"
echo "[submit-anvil-hpar-a] stage_a_time=${STAGE_A_TIME}"
echo "[submit-anvil-hpar-a] stage_a_job_name=${STAGE_A_JOB_NAME}"
echo "[submit-anvil-hpar-a] stage_a_token_position_ids=${STAGE_A_TOKEN_POSITION_IDS:-last_token}"
echo "[submit-anvil-hpar-a] stage_a_method=${STAGE_A_METHOD:-uot}"
echo "[submit-anvil-hpar-a] stage_a_hparam_selection=${STAGE_A_HPARAM_SELECTION:-rowwise}"
echo "[submit-anvil-hpar-a] stage_a_rerank_top_k=${STAGE_A_RERANK_TOP_K:-0}"
echo "[submit-anvil-hpar-a] stage_a_rerank_drop_ratio=${STAGE_A_RERANK_DROP_RATIO:-0}"
echo "[submit-anvil-hpar-a] stage_a_rerank_min_k=${STAGE_A_RERANK_MIN_K:-6}"
echo "[submit-anvil-hpar-a] stage_a_rerank_max_k=${STAGE_A_RERANK_MAX_K:-8}"
echo "[submit-anvil-hpar-a] uot_beta_neurals=${UOT_BETA_NEURALS:-0.03,0.1,0.3,1,3}"
echo "[submit-anvil-hpar-a] batch_size=${BATCH_SIZE:-64}"
echo "[submit-anvil-hpar-a] sbatch_extra_args=${SBATCH_EXTRA_ARGS}"
echo "[submit-anvil-hpar-a] split_seed=${SPLIT_SEED}"

SBATCH_ARGS=(
  --account="${ANVIL_ACCOUNT}"
  --partition="${ANVIL_PARTITION}"
  --time="${STAGE_A_TIME}"
  --export=ALL
  --job-name="${STAGE_A_JOB_NAME}"
)
if [[ -n "${SBATCH_EXTRA_ARGS}" ]]; then
  read -r -a EXTRA_ARGS <<< "${SBATCH_EXTRA_ARGS}"
  SBATCH_ARGS+=("${EXTRA_ARGS[@]}")
fi

RESULTS_TIMESTAMP="${TIMESTAMP}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
SPLIT_SEED="${SPLIT_SEED}" \
sbatch \
  "${SBATCH_ARGS[@]}" \
  scripts/anvil_mcqa_hierarchical_parallel_stage_a.sbatch

echo "[submit-anvil-hpar-a] results root will be ${RESULTS_ROOT}/${TIMESTAMP}_mcqa_hierarchical_sweep"
