#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/Users/jchang153/miniforge3/envs/torch-metal/bin/python}"
DEVICE="${DEVICE:-mps}"
RUN_NAME="${RUN_NAME:-binary_addition_h16_10seeds_local_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
RUN_DIR="${RESULTS_ROOT}/${RUN_NAME}"
SEEDS="0,1,2,3,4,5,6,7,8,9"
SOURCE_POLICY="structured_26_top3carry_c2x5_c3x7_no_random"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/causal_ot_mplconfig}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"
mkdir -p "${RUN_DIR}" "${MPLCONFIGDIR}"
"${PYTHON_BIN}" -c 'import torch; assert torch.backends.mps.is_available(), "MPS unavailable"'

run_seed_suite() {
  local seed="$1"
  local resume_flag="$2"
  local mib_extra_flag="$3"

  "${PYTHON_BIN}" experiments/binary_addition/run_single_stage_plot.py \
    --out-dir "${RUN_DIR}/single_stage" \
    --device "${DEVICE}" \
    --hidden-size 16 \
    --seeds "${seed}" \
    --basis canonical \
    --resolutions 1,2,4,8,16 \
    --fit-bases 128 \
    --calib-bases 64 \
    --test-bases 64 \
    --source-policy "${SOURCE_POLICY}" \
    ${resume_flag}

  "${PYTHON_BIN}" experiments/binary_addition/run_single_stage_plot.py \
    --out-dir "${RUN_DIR}/single_stage" \
    --device "${DEVICE}" \
    --hidden-size 16 \
    --seeds "${seed}" \
    --basis pca \
    --resolutions 1,2,4,8,16 \
    --fit-bases 128 \
    --calib-bases 64 \
    --test-bases 64 \
    --source-policy "${SOURCE_POLICY}" \
    ${resume_flag}

  "${PYTHON_BIN}" experiments/binary_addition/run_progressive_plot.py \
    --out-dir "${RUN_DIR}/progressive_ot" \
    --device "${DEVICE}" \
    --hidden-size 16 \
    --seeds "${seed}" \
    --alignment-method ot \
    --canonical-resolutions 1,2,4,8,16 \
    --pca-resolutions 1,2,4,8,16 \
    --fit-bases 128 \
    --calib-bases 64 \
    --test-bases 64 \
    --source-policy "${SOURCE_POLICY}" \
    --das-fit-bank-mode shared \
    --skip-support-guided-das \
    --exclude-stage-a-method \
    ${resume_flag}

  "${PYTHON_BIN}" experiments/binary_addition/run_progressive_plot.py \
    --out-dir "${RUN_DIR}/progressive_cosine" \
    --device "${DEVICE}" \
    --hidden-size 16 \
    --seeds "${seed}" \
    --alignment-method cosine \
    --canonical-resolutions 1,2,4,8,16 \
    --fit-bases 128 \
    --calib-bases 64 \
    --test-bases 64 \
    --source-policy "${SOURCE_POLICY}" \
    --skip-das \
    --skip-pca \
    --exclude-stage-a-method \
    ${resume_flag}

  "${PYTHON_BIN}" experiments/binary_addition/run_progressive_plot.py \
    --out-dir "${RUN_DIR}/progressive_bruteforce" \
    --device "${DEVICE}" \
    --hidden-size 16 \
    --seeds "${seed}" \
    --alignment-method brute-force \
    --canonical-resolutions 1,2,4,8,16 \
    --fit-bases 128 \
    --calib-bases 64 \
    --test-bases 64 \
    --source-policy "${SOURCE_POLICY}" \
    --skip-das \
    --skip-pca \
    --exclude-stage-a-method \
    ${resume_flag}

  "${PYTHON_BIN}" experiments/binary_addition/run_mib_baselines.py \
    --out-dir "${RUN_DIR}" \
    --run-name mib \
    --device "${DEVICE}" \
    --methods full-state,dbm-canonical,dbm-pca \
    --seeds "${seed}" \
    --rows C1,C2,C3 \
    --timesteps 0,1,2,3 \
    --width 4 \
    --hidden-size 16 \
    --fit-bases 128 \
    --calib-bases 64 \
    --test-bases 64 \
    --source-policy "${SOURCE_POLICY}" \
    --batch-size 64 \
    --eval-batch-size 512 \
    --epochs 8 \
    --learning-rate 0.01 \
    --temperature-start 1.0 \
    --temperature-end 0.01 \
    --regularization-coefficients 0,1e-5,1e-4,1e-3 \
    ${mib_extra_flag}
}

echo "run=${RUN_NAME} device=${DEVICE} python=${PYTHON_BIN} results=${RUN_DIR}"
for seed in 0 1 2 3 4 5 6 7 8 9; do
  echo "seed=${seed} started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  run_seed_suite "${seed}" "--skip-existing" "--skip-aggregate"
done

# Re-open the resume-safe artifacts with the complete seed list to replace the
# transient one-seed aggregates with definitive ten-seed summaries.
run_seed_suite "${SEEDS}" "--skip-existing" ""
"${PYTHON_BIN}" experiments/binary_addition/summarize_10seed_suite.py --run-dir "${RUN_DIR}"

echo "Final suite summary: ${RUN_DIR}/suite_summary.json"
