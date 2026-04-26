#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"
ARRAY_THROTTLE_STAGE_B="${ARRAY_THROTTLE_STAGE_B:-8}"
ARRAY_THROTTLE_STAGE_C="${ARRAY_THROTTLE_STAGE_C:-4}"
SUBMIT_FULL_DAS="${SUBMIT_FULL_DAS:-1}"

export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.secrets/hf_token")}"

ROOT="${RESULTS_ROOT}/${TIMESTAMP}_mcqa_hierarchical_sweep"
mkdir -p "${ROOT}"

echo "[submit-delta-hpar-all] timestamp=${TIMESTAMP}"
echo "[submit-delta-hpar-all] delta_account=${DELTA_ACCOUNT}"
echo "[submit-delta-hpar-all] delta_partition=${DELTA_PARTITION}"
echo "[submit-delta-hpar-all] stage_b_array_throttle=${ARRAY_THROTTLE_STAGE_B}"
echo "[submit-delta-hpar-all] stage_c_array_throttle=${ARRAY_THROTTLE_STAGE_C}"
echo "[submit-delta-hpar-all] submit_full_das=${SUBMIT_FULL_DAS}"

mapfile -t STAGE_A_JOBS < <(
  bash scripts/submit_delta_mcqa_hierarchical_parallel_stage_a.sh "${TIMESTAMP}" 2>&1 \
    | tee "${ROOT}/stage_a_submit.log" \
    | awk '/Submitted batch job/ {print $4}'
)

if [[ "${#STAGE_A_JOBS[@]}" -ne 1 ]]; then
  echo "[submit-delta-hpar-all] expected one Stage A job id, got: ${STAGE_A_JOBS[*]:-none}"
  exit 1
fi

STAGE_A_JOB="${STAGE_A_JOBS[0]}"
echo "[submit-delta-hpar-all] stage_a_job=${STAGE_A_JOB}"

STAGE_B_SUBMITTER_JOB="$(
  sbatch \
    --account="${DELTA_ACCOUNT}" \
    --partition="${DELTA_PARTITION}" \
    --dependency="afterok:${STAGE_A_JOB}" \
    --time="${ORCHESTRATOR_TIME:-00:30:00}" \
    --output="${ROOT}/mcqa-submit-b-%j.out" \
    --error="${ROOT}/mcqa-submit-b-%j.err" \
    --job-name=mcqa-submit-b \
    --wrap="cd ${PWD} && export HF_TOKEN=\"\$(cat \$HOME/.secrets/hf_token)\" && ARRAY_THROTTLE_STAGE_B=${ARRAY_THROTTLE_STAGE_B} ARRAY_THROTTLE_STAGE_C=${ARRAY_THROTTLE_STAGE_C} bash scripts/delta_mcqa_hierarchical_parallel_after_stage_a.sh ${TIMESTAMP}" \
    | awk '/Submitted batch job/ {print $4}'
)"
echo "[submit-delta-hpar-all] stage_b_submitter_job=${STAGE_B_SUBMITTER_JOB}"

if [[ "${SUBMIT_FULL_DAS}" == "1" ]]; then
  FULL_DAS_TIMESTAMP="${FULL_DAS_TIMESTAMP:-${TIMESTAMP}_full_das}"
  mapfile -t FULL_DAS_JOBS < <(
    bash scripts/submit_delta_mcqa_full_das_timed.sh "${FULL_DAS_TIMESTAMP}" 2>&1 \
      | tee "${ROOT}/full_das_submit.log" \
      | awk '/Submitted batch job/ {print $4}'
  )
  echo "[submit-delta-hpar-all] full_das_jobs=${FULL_DAS_JOBS[*]:-none}"
fi

echo "[submit-delta-hpar-all] chain submitted"
echo "[submit-delta-hpar-all] results root: ${ROOT}"
