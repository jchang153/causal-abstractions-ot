#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "No active Slurm allocation. Allocate four A40 GPUs before running this script." >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/u/${USER}/causal-abstractions-ot}"
VENV_PATH="${VENV_PATH:-/work/nvme/bgvo/${USER}/venvs/causal-ot-mcqa}"
RESULTS_ROOT="${RESULTS_ROOT:-/work/nvme/bgvo/${USER}/mcqa_pca_mib_results}"
CACHE_ROOT="${CACHE_ROOT:-/work/nvme/bgvo/${USER}/hf_cache}"
RUN_PREFIX="${RUN_PREFIX:-mcqa_pca_mib_$(date +%Y%m%d_%H%M%S)}"
PCA_NUM_BANDS_VALUES="${PCA_NUM_BANDS_VALUES:-1,2,4,8,16,32,64}"
CPUS_PER_SEED="${CPUS_PER_SEED:-8}"
ONE_SEED_SCRIPT="${REPO_ROOT}/experiments/mcqa/slurm/run_delta_mcqa_pca_mib_one_seed.sh"
LOG_ROOT="${RESULTS_ROOT}/${RUN_PREFIX}_logs"

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before starting the sweep.}"
mkdir -p "${LOG_ROOT}"

echo "[pca-mib-4seed] prefix=${RUN_PREFIX} bands=${PCA_NUM_BANDS_VALUES}"
echo "[pca-mib-4seed] launching seeds 0,1,2,3 across four one-GPU Slurm steps"

pids=()
for seed in 0 1 2 3; do
  srun \
    --exclusive \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    --cpus-per-task="${CPUS_PER_SEED}" \
    --output="${LOG_ROOT}/seed${seed}.out" \
    --error="${LOG_ROOT}/seed${seed}.err" \
    env \
      MCQA_PCA_MIB_INSIDE_SRUN=1 \
      REPO_ROOT="${REPO_ROOT}" \
      VENV_PATH="${VENV_PATH}" \
      RESULTS_ROOT="${RESULTS_ROOT}" \
      CACHE_ROOT="${CACHE_ROOT}" \
      RUN_ID="${RUN_PREFIX}_seed${seed}" \
      SPLIT_SEED="${seed}" \
      PCA_NUM_BANDS_VALUES="${PCA_NUM_BANDS_VALUES}" \
      HF_TOKEN="${HF_TOKEN}" \
      bash "${ONE_SEED_SCRIPT}" &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done

if (( failures > 0 )); then
  echo "[pca-mib-4seed] ${failures} seed run(s) failed; inspect ${LOG_ROOT}" >&2
  exit 1
fi

echo "[pca-mib-4seed] all four seeds complete"
echo "[pca-mib-4seed] results=${RESULTS_ROOT} logs=${LOG_ROOT}"
