#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
SPLIT_SEED="${SPLIT_SEED:-0}"

DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-8}"
FULL_DAS_LAYER_TIME="${FULL_DAS_LAYER_TIME:-02:00:00}"
LAYER_INDICES="${LAYER_INDICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}"

LAYER_COUNT="$(
  python -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x.strip()]))' "${LAYER_INDICES}"
)"
if [[ "${LAYER_COUNT}" -le 0 ]]; then
  echo "[submit-delta-full-das-layer] no layers requested"
  exit 1
fi

SWEEP_ROOT="${RESULTS_ROOT}/${TIMESTAMP}_full_das_layer_sweep"
mkdir -p "${SWEEP_ROOT}"

echo "[submit-delta-full-das-layer] timestamp=${TIMESTAMP}"
echo "[submit-delta-full-das-layer] delta_account=${DELTA_ACCOUNT}"
echo "[submit-delta-full-das-layer] delta_partition=${DELTA_PARTITION}"
echo "[submit-delta-full-das-layer] split_seed=${SPLIT_SEED}"
echo "[submit-delta-full-das-layer] layers=${LAYER_INDICES}"
echo "[submit-delta-full-das-layer] layer_count=${LAYER_COUNT}"
echo "[submit-delta-full-das-layer] array_throttle=${ARRAY_THROTTLE}"
echo "[submit-delta-full-das-layer] time_limit=${FULL_DAS_LAYER_TIME}"
echo "[submit-delta-full-das-layer] token_position_ids=${TOKEN_POSITION_IDS:-last_token}"
echo "[submit-delta-full-das-layer] das_subspace_dims=${DAS_SUBSPACE_DIMS:-32,64,96,128,256,512,768,1024,1536,2048,2304}"

RESULTS_TIMESTAMP="${TIMESTAMP}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
SPLIT_SEED="${SPLIT_SEED}" \
LAYER_INDICES="${LAYER_INDICES}" \
sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --array="0-$((LAYER_COUNT - 1))%${ARRAY_THROTTLE}" \
  --time="${FULL_DAS_LAYER_TIME}" \
  --export=ALL \
  --job-name=mcqa-full-das-layer \
  --output="${SWEEP_ROOT}/mcqa-full-das-layer-%A_%a.out" \
  --error="${SWEEP_ROOT}/mcqa-full-das-layer-%A_%a.err" \
  scripts/delta_mcqa_full_das_layer_task.sbatch

echo "[submit-delta-full-das-layer] results root will be ${SWEEP_ROOT}"
echo "[submit-delta-full-das-layer] aggregate after completion with:"
echo "  python mcqa_full_das_layer_aggregate.py ${SWEEP_ROOT}"
