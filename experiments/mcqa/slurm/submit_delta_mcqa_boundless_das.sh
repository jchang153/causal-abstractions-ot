#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA40x4}"

echo "[submit-mcqa-bdas] timestamp=${TIMESTAMP} results_root=${RESULTS_ROOT}"
RESULTS_TIMESTAMP="${TIMESTAMP}" RESULTS_ROOT="${RESULTS_ROOT}" sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --export=ALL \
  experiments/mcqa/slurm/delta_mcqa_boundless_das.sbatch
