#!/usr/bin/env bash
set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA40x4}"
RUN_NAME="${RUN_NAME:-binary_addition_h16_10seeds_${TIMESTAMP}}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"

echo "[submit-binadd] run=${RUN_NAME} results_root=${RESULTS_ROOT}"
RUN_NAME="${RUN_NAME}" RESULTS_ROOT="${RESULTS_ROOT}" sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --export=ALL \
  experiments/binary_addition/slurm/delta_binary_addition_10seed_suite.sbatch
