#!/bin/bash
#SBATCH --job-name=mcqa-block-a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module purge
module load anaconda

VENV_PATH="${VENV_PATH:-${HOME}/envs/causal-ot}"
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

export RESULTS_ROOT="${RESULTS_ROOT:-results/anvil}"
export RESULTS_TIMESTAMP="${RESULTS_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export LAYER_BLOCKS="${LAYER_BLOCKS:-0-4;5-9;10-14;15-19;20-25}"
export TOKEN_POSITION_IDS="${TOKEN_POSITION_IDS:-last_token}"
export TARGET_VARS="${TARGET_VARS:-answer_pointer,answer_token}"
export SIGNATURE_MODE="${SIGNATURE_MODE:-family_label_delta_norm}"
export OT_EPSILONS="${OT_EPSILONS:-0.5,1,2,4}"
export OT_TOP_K_VALUES="${OT_TOP_K_VALUES:-1,2,4}"
export OT_LAMBDAS="${OT_LAMBDAS:-0.5,1,2,4}"
export CALIBRATION_METRIC="${CALIBRATION_METRIC:-family_weighted_macro_exact_acc}"
export CALIBRATION_FAMILY_WEIGHTS="${CALIBRATION_FAMILY_WEIGHTS:-1,1.5,2}"
export SPLIT_SEED="${SPLIT_SEED:-0}"

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

echo "[anvil-mcqa-block-a] host=$(hostname) timestamp=${RESULTS_TIMESTAMP} split_seed=${SPLIT_SEED}"
echo "[anvil-mcqa-block-a] layer_blocks=${LAYER_BLOCKS}"
echo "[anvil-mcqa-block-a] token_position_ids=${TOKEN_POSITION_IDS} signature_mode=${SIGNATURE_MODE}"
nvidia-smi

ARGS=(
  --preset next
  --device cuda
  --dataset-path jchang153/copycolors_mcqa
  --results-root "${RESULTS_ROOT}"
  --results-timestamp "${RESULTS_TIMESTAMP}"
  --signatures-dir signatures
  --methods ot
  --target-vars "${TARGET_VARS}"
  --layers auto
  --layer-blocks "${LAYER_BLOCKS}"
  --token-position-ids "${TOKEN_POSITION_IDS}"
  --resolutions full
  --signature-modes "${SIGNATURE_MODE}"
  --ot-epsilons "${OT_EPSILONS}"
  --ot-top-k-values "${OT_TOP_K_VALUES}"
  --ot-lambdas "${OT_LAMBDAS}"
  --calibration-metric "${CALIBRATION_METRIC}"
  --calibration-family-weights "${CALIBRATION_FAMILY_WEIGHTS}"
  --split-seed "${SPLIT_SEED}"
)

srun --unbuffered python mcqa_run_cloud.py "${ARGS[@]}"
