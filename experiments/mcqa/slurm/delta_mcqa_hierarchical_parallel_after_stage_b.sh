#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:?Usage: bash experiments/mcqa/slurm/delta_mcqa_hierarchical_parallel_after_stage_b.sh <timestamp>}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"
ARRAY_THROTTLE_STAGE_C="${ARRAY_THROTTLE_STAGE_C:-${ARRAY_THROTTLE:-4}}"
SPLIT_SEED="${SPLIT_SEED:-0}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.secrets/hf_token")}"

ROOT="${RESULTS_ROOT}/${TIMESTAMP}_mcqa_hierarchical_sweep"
mkdir -p "${ROOT}"

echo "[orchestrate-after-b] submitting Stage C arrays for timestamp=${TIMESTAMP} split_seed=${SPLIT_SEED}"
mapfile -t STAGE_C_JOBS < <(
  SPLIT_SEED="${SPLIT_SEED}" ARRAY_THROTTLE="${ARRAY_THROTTLE_STAGE_C}" \
    bash experiments/mcqa/slurm/submit_delta_mcqa_hierarchical_parallel_stage_c.sh "${TIMESTAMP}" 2>&1 \
    | tee "${ROOT}/stage_c_submit.log" \
    | awk '/Submitted batch job/ {print $4}'
)

if [[ "${#STAGE_C_JOBS[@]}" -eq 0 ]]; then
  echo "[orchestrate-after-b] no Stage C job ids captured; submitting aggregation without Stage C dependency"
  AGG_DEPENDENCY_ARGS=()
else
  STAGE_C_DEPENDENCY="afterok:$(IFS=:; echo "${STAGE_C_JOBS[*]}")"
  echo "[orchestrate-after-b] stage_c_jobs=${STAGE_C_JOBS[*]}"
  echo "[orchestrate-after-b] submitting aggregation dependency=${STAGE_C_DEPENDENCY}"
  AGG_DEPENDENCY_ARGS=(--dependency="${STAGE_C_DEPENDENCY}")
fi

sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${ORCHESTRATOR_CPUS_PER_TASK:-1}" \
  --gpus-per-node="${ORCHESTRATOR_GPUS_PER_NODE:-1}" \
  --mem="${ORCHESTRATOR_MEM:-8g}" \
  --constraint="${DELTA_CONSTRAINT:-scratch}" \
  "${AGG_DEPENDENCY_ARGS[@]}" \
  --time="${ORCHESTRATOR_TIME:-00:30:00}" \
  --output="${ROOT}/mcqa-agg-%j.out" \
  --error="${ROOT}/mcqa-agg-%j.err" \
  --job-name=mcqa-agg \
  --wrap="cd ${PWD} && bash experiments/mcqa/slurm/aggregate_delta_mcqa_hierarchical_parallel.sh ${TIMESTAMP}"
