#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/Users/jchang153/miniforge3/envs/torch-metal/bin/python}"
DEVICE="cpu"
SEED="0"
WIDTH="8"
HIDDEN_SIZE="64"
ROWS="C1,C2,C3,C4,C5,C6,C7"
RESOLUTIONS="1,2,4,8,16,32,64"
SOURCE_POLICY="structured_top3carry_c2x5_c3x7_no_random"
RUN_NAME="${RUN_NAME:-binary_addition_w8_h64_plot4_seed0_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
RUN_DIR="${RESULTS_ROOT}/${RUN_NAME}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoint_seed0"
CHECKPOINT="${CHECKPOINT_DIR}/gru_adder.pt"
TRAIN_SUMMARY="${CHECKPOINT_DIR}/train_summary.json"
CHECKPOINT_MAP="0=${CHECKPOINT}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/causal_ot_mplconfig}"
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"
mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}" "${MPLCONFIGDIR}"
"${PYTHON_BIN}" -c 'import torch; print(f"torch={torch.__version__} device=cpu")'

checkpoint_is_valid() {
  [[ -f "${CHECKPOINT}" && -f "${TRAIN_SUMMARY}" ]] || return 1
  "${PYTHON_BIN}" -c '
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {"all": 1.0, "fit": 1.0, "calib": 1.0, "test": 1.0}
assert summary.get("factual_exact") == expected
assert summary.get("split_sizes") == {"fit": 512, "calib": 256, "test": 256}
' "${TRAIN_SUMMARY}"
}

if ! checkpoint_is_valid; then
  "${PYTHON_BIN}" experiments/binary_addition/run_train_backbone.py \
    --out-dir "${CHECKPOINT_DIR}" \
    --width "${WIDTH}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --batch-size 512 \
    --eval-batch-size 4096 \
    --epochs 250 \
    --learning-rate 0.01 \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --fit-bases 512 \
    --calib-bases 256 \
    --test-bases 256 \
    --train-on all
fi
checkpoint_is_valid

COMMON_ARGS=(
  --device "${DEVICE}"
  --hidden-size "${HIDDEN_SIZE}"
  --width "${WIDTH}"
  --rows "${ROWS}"
  --seeds "${SEED}"
  --checkpoint-map "${CHECKPOINT_MAP}"
  --fit-bases 512
  --calib-bases 256
  --test-bases 256
  --train-on all
  --train-epochs 250
  --train-batch-size 512
  --train-lr 0.01
  --source-policy "${SOURCE_POLICY}"
  --cache-batch-size 1024
  --skip-existing
)

"${PYTHON_BIN}" experiments/binary_addition/run_single_stage_plot.py \
  --out-dir "${RUN_DIR}/single_stage" \
  "${COMMON_ARGS[@]}" \
  --basis canonical \
  --resolutions "${RESOLUTIONS}" \
  --batch-size 512

"${PYTHON_BIN}" experiments/binary_addition/run_single_stage_plot.py \
  --out-dir "${RUN_DIR}/single_stage" \
  "${COMMON_ARGS[@]}" \
  --basis pca \
  --resolutions "${RESOLUTIONS}" \
  --batch-size 512

"${PYTHON_BIN}" experiments/binary_addition/run_progressive_plot.py \
  --out-dir "${RUN_DIR}/progressive_ot" \
  "${COMMON_ARGS[@]}" \
  --alignment-method ot \
  --canonical-resolutions "${RESOLUTIONS}" \
  --pca-resolutions "${RESOLUTIONS}" \
  --das-batch-size 512 \
  --skip-das \
  --skip-support-guided-das \
  --exclude-stage-a-method

"${PYTHON_BIN}" experiments/binary_addition/summarize_plot4_pilot.py \
  --run-dir "${RUN_DIR}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --seed "${SEED}"

echo "Final pilot summary: ${RUN_DIR}/plot4_summary.json"
