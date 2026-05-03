#!/bin/bash
#SBATCH --job-name=mcqa-uot-a
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
export TOKEN_POSITION_IDS="${TOKEN_POSITION_IDS:-last_token}"
export TARGET_VARS="${TARGET_VARS:-answer_pointer,answer_token}"
export OT_SOURCE_TARGET_VARS="${OT_SOURCE_TARGET_VARS:-${TARGET_VARS}}"
export SIGNATURE_MODE="${SIGNATURE_MODE:-family_label_delta_norm}"
export OT_EPSILONS="${OT_EPSILONS:-0.5,1,2,4}"
export UOT_BETA_NEURALS="${UOT_BETA_NEURALS:-0.03,0.1,0.3,1,3}"
export OT_TOP_K_VALUES="${OT_TOP_K_VALUES:-1,2,4}"
export OT_LAMBDAS="${OT_LAMBDAS:-0.5,1,2,4}"
export CALIBRATION_METRIC="${CALIBRATION_METRIC:-family_weighted_macro_exact_acc}"
export CALIBRATION_FAMILY_WEIGHTS="${CALIBRATION_FAMILY_WEIGHTS:-1,1.5,2}"
export SPLIT_SEED="${SPLIT_SEED:-0}"

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

echo "[delta-mcqa-uot-a] host=$(hostname) timestamp=${RESULTS_TIMESTAMP} split_seed=${SPLIT_SEED}"
echo "[delta-mcqa-uot-a] token_position_ids=${TOKEN_POSITION_IDS} signature_mode=${SIGNATURE_MODE}"
echo "[delta-mcqa-uot-a] target_vars=${TARGET_VARS} ot_source_target_vars=${OT_SOURCE_TARGET_VARS}"
echo "[delta-mcqa-uot-a] ot_epsilons=${OT_EPSILONS} uot_beta_neurals=${UOT_BETA_NEURALS}"
nvidia-smi

ARGS=(
  --preset next
  --device cuda
  --dataset-path jchang153/copycolors_mcqa
  --results-root "${RESULTS_ROOT}"
  --results-timestamp "${RESULTS_TIMESTAMP}"
  --signatures-dir signatures
  --methods uot
  --target-vars "${TARGET_VARS}"
  --ot-source-target-vars "${OT_SOURCE_TARGET_VARS}"
  --layers auto
  --token-position-ids "${TOKEN_POSITION_IDS}"
  --resolutions full
  --signature-modes "${SIGNATURE_MODE}"
  --ot-epsilons "${OT_EPSILONS}"
  --uot-beta-neural "${UOT_BETA_NEURALS}"
  --ot-top-k-values "${OT_TOP_K_VALUES}"
  --ot-lambdas "${OT_LAMBDAS}"
  --calibration-metric "${CALIBRATION_METRIC}"
  --calibration-family-weights "${CALIBRATION_FAMILY_WEIGHTS}"
  --split-seed "${SPLIT_SEED}"
)

srun --unbuffered python mcqa_run_cloud.py "${ARGS[@]}"
