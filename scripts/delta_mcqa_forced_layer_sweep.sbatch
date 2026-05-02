#!/bin/bash
#SBATCH --job-name=mcqa-force
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --constraint="scratch"
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

VENV_PATH="${VENV_PATH:-${HOME}/venvs/causal-ot}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing virtual environment at ${VENV_PATH}"
  exit 1
fi
source "${VENV_PATH}/bin/activate"

export HF_TOKEN="${HF_TOKEN:-$(cat "${HOME}/.secrets/hf_token")}"
export HF_HOME="${SCRATCH:-/tmp/${USER}}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
export RESULTS_TIMESTAMP="${RESULTS_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export FORCED_LAYER_INDICES="${FORCED_LAYER_INDICES:-17,18,19,23,24}"
export TOKEN_POSITION_IDS="${TOKEN_POSITION_IDS:-last_token}"
export TARGET_VARS="${TARGET_VARS:-answer_pointer,answer_token}"
export SIGNATURE_MODE="${SIGNATURE_MODE:-family_label_delta_norm}"
export OT_LAMBDAS="${OT_LAMBDAS:-0.5,1,2,4}"
export CALIBRATION_METRIC="${CALIBRATION_METRIC:-family_weighted_macro_exact_acc}"
export CALIBRATION_FAMILY_WEIGHTS="${CALIBRATION_FAMILY_WEIGHTS:-1,1.5,2}"
export SPLIT_SEED="${SPLIT_SEED:-0}"

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

echo "[delta-mcqa-force] host=$(hostname) timestamp=${RESULTS_TIMESTAMP} split_seed=${SPLIT_SEED}"
echo "[delta-mcqa-force] forced_layer_indices=${FORCED_LAYER_INDICES}"
echo "[delta-mcqa-force] token_position_ids=${TOKEN_POSITION_IDS} targets=${TARGET_VARS}"
nvidia-smi

ARGS=(
  --preset next
  --device cuda
  --dataset-path jchang153/copycolors_mcqa
  --results-root "${RESULTS_ROOT}"
  --results-timestamp "${RESULTS_TIMESTAMP}"
  --signatures-dir signatures
  --methods bruteforce
  --target-vars "${TARGET_VARS}"
  --layer-sweep
  --layer-indices "${FORCED_LAYER_INDICES}"
  --token-position-ids "${TOKEN_POSITION_IDS}"
  --resolutions full
  --signature-modes "${SIGNATURE_MODE}"
  --ot-lambdas "${OT_LAMBDAS}"
  --calibration-metric "${CALIBRATION_METRIC}"
  --calibration-family-weights "${CALIBRATION_FAMILY_WEIGHTS}"
  --split-seed "${SPLIT_SEED}"
)

srun --unbuffered python mcqa_run_cloud.py "${ARGS[@]}"
