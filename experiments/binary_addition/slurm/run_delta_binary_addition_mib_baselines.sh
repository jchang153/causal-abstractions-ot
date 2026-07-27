#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "No active Slurm allocation. Run this launcher inside salloc." >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/u/${USER}/causal-abstractions-ot}"
VENV_PATH="${VENV_PATH:-/work/nvme/bgvo/${USER}/venvs/causal-ot-py311}"
RUN_NAME="${RUN_NAME:-binary_addition_mib_baselines_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/delta}"

cd "${REPO_ROOT}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Virtual environment not found at ${VENV_PATH}" >&2
  exit 2
fi
source "${VENV_PATH}/bin/activate"
python -c 'import sys; assert sys.version_info >= (3, 10), sys.version'

echo "job=${SLURM_JOB_ID} run=${RUN_NAME} results=${RESULTS_ROOT}/${RUN_NAME}"

# One GPU and one outer Slurm step for the complete resume-safe sweep.
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --gpus-per-task=1 --gpu-bind=single:1 \
  python experiments/binary_addition/run_mib_baselines.py \
    --device cuda \
    --methods full-state,dbm-canonical,dbm-pca \
    --seeds 0,1,2 \
    --rows C1,C2,C3 \
    --timesteps 0,1,2,3 \
    --width 4 \
    --hidden-size 16 \
    --fit-bases 128 \
    --calib-bases 64 \
    --test-bases 64 \
    --source-policy structured_26_top3carry_c2x5_c3x7_no_random \
    --batch-size 64 \
    --eval-batch-size 512 \
    --epochs 8 \
    --learning-rate 0.01 \
    --temperature-start 1.0 \
    --temperature-end 0.01 \
    --regularization-coefficient 0.0 \
    --out-dir "${RESULTS_ROOT}" \
    --run-name "${RUN_NAME}"

echo "All outputs: ${RESULTS_ROOT}/${RUN_NAME}"
