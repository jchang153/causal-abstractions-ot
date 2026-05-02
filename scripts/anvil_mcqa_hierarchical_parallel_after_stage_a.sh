#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:?Usage: bash scripts/anvil_mcqa_hierarchical_parallel_after_stage_a.sh <timestamp>}"
RESULTS_ROOT="${RESULTS_ROOT:-results/anvil}"
ANVIL_ACCOUNT="${ANVIL_ACCOUNT:-cis260602-ai}"
ANVIL_PARTITION="${ANVIL_PARTITION:-ai}"
ARRAY_THROTTLE_STAGE_B="${ARRAY_THROTTLE_STAGE_B:-${ARRAY_THROTTLE:-8}}"
SPLIT_SEED="${SPLIT_SEED:-0}"
SUBMIT_STAGE_C="${SUBMIT_STAGE_C:-1}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.secrets/hf_token")}"

ROOT="${RESULTS_ROOT}/${TIMESTAMP}_mcqa_hierarchical_sweep"
mkdir -p "${ROOT}"

echo "[orchestrate-after-a] submitting Stage B arrays for timestamp=${TIMESTAMP} split_seed=${SPLIT_SEED}"
mapfile -t STAGE_B_JOBS < <(
  SPLIT_SEED="${SPLIT_SEED}" ARRAY_THROTTLE="${ARRAY_THROTTLE_STAGE_B}" \
    bash scripts/submit_anvil_mcqa_hierarchical_parallel_stage_b.sh "${TIMESTAMP}" 2>&1 \
    | tee "${ROOT}/stage_b_submit.log" \
    | awk '/Submitted batch job/ {print $4}'
)

if [[ "${#STAGE_B_JOBS[@]}" -eq 0 ]]; then
  echo "[orchestrate-after-a] no Stage B job ids captured"
  exit 1
fi

STAGE_B_DEPENDENCY="afterok:$(IFS=:; echo "${STAGE_B_JOBS[*]}")"
echo "[orchestrate-after-a] stage_b_jobs=${STAGE_B_JOBS[*]}"
echo "[orchestrate-after-a] submit_stage_c=${SUBMIT_STAGE_C}"
if [[ "${SUBMIT_STAGE_C}" != "1" ]]; then
  echo "[orchestrate-after-a] skipping Stage C submitter; submit aggregation after Stage B dependency=${STAGE_B_DEPENDENCY}"
  sbatch \
    --account="${ANVIL_ACCOUNT}" \
    --partition="${ANVIL_PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${ORCHESTRATOR_CPUS_PER_TASK:-1}" \
    --gpus-per-node="${ORCHESTRATOR_GPUS_PER_NODE:-1}" \
    --mem="${ORCHESTRATOR_MEM:-8g}" \
    --dependency="${STAGE_B_DEPENDENCY}" \
    --time="${ORCHESTRATOR_TIME:-00:30:00}" \
    --output="${ROOT}/mcqa-agg-%j.out" \
    --error="${ROOT}/mcqa-agg-%j.err" \
    --job-name=mcqa-agg \
    --wrap="cd ${PWD} && ALLOW_PARTIAL=1 bash scripts/aggregate_anvil_mcqa_hierarchical_parallel.sh ${TIMESTAMP}"
  exit 0
fi

echo "[orchestrate-after-a] submitting Stage C submitter dependency=${STAGE_B_DEPENDENCY}"

sbatch \
  --account="${ANVIL_ACCOUNT}" \
  --partition="${ANVIL_PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${ORCHESTRATOR_CPUS_PER_TASK:-1}" \
  --gpus-per-node="${ORCHESTRATOR_GPUS_PER_NODE:-1}" \
  --mem="${ORCHESTRATOR_MEM:-8g}" \
  --dependency="${STAGE_B_DEPENDENCY}" \
  --time="${ORCHESTRATOR_TIME:-00:30:00}" \
  --output="${ROOT}/mcqa-submit-c-%j.out" \
  --error="${ROOT}/mcqa-submit-c-%j.err" \
  --job-name=mcqa-submit-c \
  --wrap="cd ${PWD} && export HF_TOKEN=\"\$(cat \$HOME/.secrets/hf_token)\" && SPLIT_SEED=${SPLIT_SEED} bash scripts/anvil_mcqa_hierarchical_parallel_after_stage_b.sh ${TIMESTAMP}"
