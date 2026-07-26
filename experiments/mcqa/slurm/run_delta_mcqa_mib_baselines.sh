#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "No active Slurm allocation. First run:" >&2
  echo "salloc --account=bgvo-delta-gpu --partition=gpuA40x4 --gres=gpu:nvidia_a40:1 --cpus-per-task=8 --mem=64G --time=2-00:00:00" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/u/${USER}/causal-abstractions-ot}"
RUN_NAME="${RUN_NAME:-mcqa_mib_baselines_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/delta}"
CACHE_ROOT="${CACHE_ROOT:-/work/nvme/bgvo/${USER}/hf_cache}"

mkdir -p "${RESULTS_ROOT}" "${CACHE_ROOT}/hub" "${CACHE_ROOT}/datasets" "${CACHE_ROOT}/transformers"

export HF_HOME="${CACHE_ROOT}"
export HF_HUB_CACHE="${CACHE_ROOT}/hub"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd "${REPO_ROOT}"
source /u/${USER}/.venv/bin/activate

echo "job=${SLURM_JOB_ID} run=${RUN_NAME} cache=${CACHE_ROOT} results=${RESULTS_ROOT}/${RUN_NAME}"

run_baseline_shard() {
  python experiments/mcqa/mcqa_dbm_baselines.py \
    --device cuda \
    --methods full-layer,dbm-canonical,dbm-pca,dbm-sae \
    --targets answer_pointer,answer_token \
    --layers all \
    --seeds 0,1,2 \
    --train-size 200 \
    --calibration-size 200 \
    --test-size 200 \
    --batch-size 64 \
    --eval-batch-size 128 \
    --epochs 8 \
    --learning-rate 0.01 \
    --temperature-start 1.0 \
    --temperature-end 0.01 \
    --regularization-coefficient 0.0 \
    --sae-release gemma-scope-2b-pt-res-canonical \
    --sae-id-template 'layer_{layer}/width_16k/canonical' \
    --results-root "${RESULTS_ROOT}" \
    --run-name "${RUN_NAME}" \
    --num-shards "${SLURM_NTASKS:-1}" \
    --shard-index "${SLURM_PROCID:-0}"
}

if [[ "${MCQA_MIB_INSIDE_SRUN:-0}" == "1" ]]; then
  run_baseline_shard
  exit 0
fi

# One Slurm step owns the entire resumable sweep. The prior allocation contains
# one A40 GPU, eight CPUs, and 64 GB of host memory, so the default is one task.
# If four GPUs were actually requested, set BASELINE_WORKERS=4; this remains one
# srun, with four disjoint, resume-safe shards.
BASELINE_WORKERS="${BASELINE_WORKERS:-1}"
CPUS_PER_WORKER="${CPUS_PER_WORKER:-$(( ${SLURM_CPUS_ON_NODE:-8} / BASELINE_WORKERS ))}"
srun --ntasks="${BASELINE_WORKERS}" --gpus-per-task=1 --gpu-bind=single:1 \
  --cpus-per-task="${CPUS_PER_WORKER}" \
  env MCQA_MIB_INSIDE_SRUN=1 REPO_ROOT="${REPO_ROOT}" RUN_NAME="${RUN_NAME}" \
    RESULTS_ROOT="${RESULTS_ROOT}" CACHE_ROOT="${CACHE_ROOT}" bash "$0"

echo "All outputs: ${RESULTS_ROOT}/${RUN_NAME}"
