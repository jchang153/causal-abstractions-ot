#!/bin/bash
#SBATCH --job-name=mcqa-bdas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=48:00:00
#SBATCH --constraint="scratch"
#SBATCH --array=0-2
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --requeue

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
VENV_PATH="${VENV_PATH:-${HOME}/venvs/causal-ot}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing virtual environment at ${VENV_PATH}" >&2
  exit 1
fi
source "${VENV_PATH}/bin/activate"

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before submitting the job.}"
export HF_HOME="${SCRATCH:-/tmp/${USER}}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
RESULTS_TIMESTAMP="${RESULTS_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SPLIT_SEED="${SLURM_ARRAY_TASK_ID}"

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"
nvidia-smi

srun --unbuffered python experiments/mcqa/mcqa_boundless_das.py \
  --device cuda \
  --model-name google/gemma-2-2b \
  --dataset-path jchang153/copycolors_mcqa \
  --dataset-size 2000 \
  --split-seed "${SPLIT_SEED}" \
  --train-pool-size 200 \
  --calibration-pool-size 200 \
  --test-pool-size 200 \
  --batch-size 64 \
  --layers all \
  --token-position-id last_token \
  --target-vars answer_pointer,answer_token \
  --epochs 12 \
  --min-epochs 5 \
  --plateau-patience 1 \
  --plateau-rel-delta 0.001 \
  --rotation-learning-rate 0.01 \
  --boundary-learning-rate 0.0001 \
  --boundary-init 0.5 \
  --boundary-penalty 1.0 \
  --temperature-start 1.0 \
  --temperature-end 0.1 \
  --restarts 1 \
  --seed 42 \
  --results-root "${RESULTS_ROOT}" \
  --results-timestamp "${RESULTS_TIMESTAMP}_seed${SPLIT_SEED}" \
  --signatures-dir signatures \
  --resume
