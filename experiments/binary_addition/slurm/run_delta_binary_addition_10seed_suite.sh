#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "No active Slurm allocation. Run this launcher inside salloc." >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/u/${USER}/causal-abstractions-ot}"
VENV_PATH="${VENV_PATH:-/work/nvme/bgvo/${USER}/venvs/causal-ot-py311}"
RUN_NAME="${RUN_NAME:-binary_addition_h16_10seeds_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/delta}"
RUN_DIR="${RESULTS_ROOT}/${RUN_NAME}"
SEEDS="0,1,2,3,4,5,6,7,8,9"
SOURCE_POLICY="structured_26_top3carry_c2x5_c3x7_no_random"

export REPO_ROOT VENV_PATH RUN_NAME RESULTS_ROOT RUN_DIR SEEDS SOURCE_POLICY

cd "${REPO_ROOT}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Virtual environment not found at ${VENV_PATH}" >&2
  exit 2
fi
source "${VENV_PATH}/bin/activate"
python -c 'import sys; assert sys.version_info >= (3, 10), sys.version'

run_seed_suite() {
  local seed="$1"
  local resume_flag="$2"
  local mib_extra_flag="$3"

  python experiments/binary_addition/run_single_stage_plot.py \
    --out-dir "${RUN_DIR}/single_stage" \
    --device cuda \
    --hidden-size 16 \
    --seeds "${seed}" \
    --basis canonical \
    --resolutions 1,2,4,8,16 \
    --fit-bases 128 \
    --calib-bases 64 \
    --test-bases 64 \
    --source-policy "${SOURCE_POLICY}" \
    ${resume_flag}

  python experiments/binary_addition/run_single_stage_plot.py \
    --out-dir "${RUN_DIR}/single_stage" \
    --device cuda \
    --hidden-size 16 \
    --seeds "${seed}" \
    --basis pca \
    --resolutions 1,2,4,8,16 \
    --fit-bases 128 \
    --calib-bases 64 \
    --test-bases 64 \
    --source-policy "${SOURCE_POLICY}" \
    ${resume_flag}

  python experiments/binary_addition/run_progressive_plot.py \
    --out-dir "${RUN_DIR}/progressive_ot" \
    --device cuda \
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

  python experiments/binary_addition/run_progressive_plot.py \
    --out-dir "${RUN_DIR}/progressive_cosine" \
    --device cuda \
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

  python experiments/binary_addition/run_progressive_plot.py \
    --out-dir "${RUN_DIR}/progressive_bruteforce" \
    --device cuda \
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

  python experiments/binary_addition/run_mib_baselines.py \
    --out-dir "${RUN_DIR}" \
    --run-name mib \
    --device cuda \
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

if [[ "${1:-}" == "--worker" ]]; then
  rank="${SLURM_PROCID:?missing SLURM_PROCID}"
  workers="${SLURM_NTASKS:?missing SLURM_NTASKS}"
  for ((seed=rank; seed<10; seed+=workers)); do
    echo "worker=${rank} seed=${seed} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
    run_seed_suite "${seed}" "--skip-existing" "--skip-aggregate"
  done
  exit 0
fi

if [[ "${1:-}" == "--aggregate" ]]; then
  run_seed_suite "${SEEDS}" "--skip-existing" ""
  python experiments/binary_addition/summarize_10seed_suite.py --run-dir "${RUN_DIR}"
  exit 0
fi

WORKERS="${WORKERS:-4}"
mkdir -p "${RUN_DIR}/logs"
echo "job=${SLURM_JOB_ID} run=${RUN_NAME} workers=${WORKERS} results=${RUN_DIR}"

srun --ntasks="${WORKERS}" \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  --output="${RUN_DIR}/logs/worker_%t.out" \
  --error="${RUN_DIR}/logs/worker_%t.err" \
  "${BASH_SOURCE[0]}" --worker

# Re-open the resume-safe outputs once with the full seed list to replace the
# transient per-worker aggregate files with the definitive 10-seed summaries.
srun --ntasks=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  "${BASH_SOURCE[0]}" --aggregate

echo "Final suite summary: ${RUN_DIR}/suite_summary.json"
