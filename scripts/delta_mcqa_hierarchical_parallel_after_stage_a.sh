#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:?Usage: bash scripts/delta_mcqa_hierarchical_parallel_after_stage_a.sh <timestamp>}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"
ARRAY_THROTTLE_STAGE_B="${ARRAY_THROTTLE_STAGE_B:-${ARRAY_THROTTLE:-8}}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.secrets/hf_token")}"

ROOT="${RESULTS_ROOT}/${TIMESTAMP}_mcqa_hierarchical_sweep"
mkdir -p "${ROOT}"

echo "[orchestrate-after-a] submitting Stage B arrays for timestamp=${TIMESTAMP}"
mapfile -t STAGE_B_JOBS < <(
  ARRAY_THROTTLE="${ARRAY_THROTTLE_STAGE_B}" \
    bash scripts/submit_delta_mcqa_hierarchical_parallel_stage_b.sh "${TIMESTAMP}" 2>&1 \
    | tee "${ROOT}/stage_b_submit.log" \
    | awk '/Submitted batch job/ {print $4}'
)

if [[ "${#STAGE_B_JOBS[@]}" -eq 0 ]]; then
  echo "[orchestrate-after-a] no Stage B job ids captured"
  exit 1
fi

STAGE_B_DEPENDENCY="afterok:$(IFS=:; echo "${STAGE_B_JOBS[*]}")"
echo "[orchestrate-after-a] stage_b_jobs=${STAGE_B_JOBS[*]}"
echo "[orchestrate-after-a] submitting Stage C submitter dependency=${STAGE_B_DEPENDENCY}"

sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --dependency="${STAGE_B_DEPENDENCY}" \
  --time="${ORCHESTRATOR_TIME:-00:30:00}" \
  --output="${ROOT}/mcqa-submit-c-%j.out" \
  --error="${ROOT}/mcqa-submit-c-%j.err" \
  --job-name=mcqa-submit-c \
  --wrap="cd ${PWD} && export HF_TOKEN=\"\$(cat \$HOME/.secrets/hf_token)\" && bash scripts/delta_mcqa_hierarchical_parallel_after_stage_b.sh ${TIMESTAMP}"
