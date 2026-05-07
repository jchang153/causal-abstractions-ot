#!/bin/bash
#SBATCH --job-name=mcqa-hpar-task
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=24:00:00
#SBATCH --constraint="scratch"
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --requeue

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

VENV_PATH="${VENV_PATH:-${HOME}/venvs/causal-ot}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing virtual environment at ${VENV_PATH}"
  exit 1
fi
source "${VENV_PATH}/bin/activate"

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before submitting the job.}"
export HF_HOME="${SCRATCH:-/tmp/${USER}}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

TASK_FILE="${TASK_FILE:?Set TASK_FILE to a hierarchical parallel task manifest.}"
TASK_INDEX="${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${TASK_INDEX}" ]]; then
  echo "TASK_INDEX or SLURM_ARRAY_TASK_ID is required."
  exit 1
fi

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

echo "[delta-mcqa-hpar-task] host=$(hostname) task_file=${TASK_FILE} task_index=${TASK_INDEX}"
nvidia-smi

srun --unbuffered python experiments/mcqa/mcqa_delta_hierarchical_parallel.py run-task \
  --task-file "${TASK_FILE}" \
  --task-index "${TASK_INDEX}"
