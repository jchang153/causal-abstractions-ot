#!/bin/bash
#SBATCH --job-name=mcqa-ot-pca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --constraint="scratch"
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --requeue

set -euo pipefail

if [[ -z "${LAYER:-}" ]]; then
  echo "LAYER is required. Submit with LAYER=<idx>."
  exit 1
fi

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
export RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
export RESULTS_TIMESTAMP="${RESULTS_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export TOKEN_POSITION_ID="${TOKEN_POSITION_ID:-last_token}"
export SITE_MENU="${SITE_MENU:-partition}"
export NUM_BANDS="${NUM_BANDS:-8}"
export BAND_SCHEME="${BAND_SCHEME:-equal}"
export TOP_PREFIX_SIZES="${TOP_PREFIX_SIZES:-8,16,32,64}"
export BASIS_SOURCE_MODE="${BASIS_SOURCE_MODE:-all_variants}"
export SCREEN_DAS="${SCREEN_DAS:-0}"
export SCREEN_MASK_NAMES="${SCREEN_MASK_NAMES:-Top1,Top2}"
export SCREEN_MAX_EPOCHS="${SCREEN_MAX_EPOCHS:-25}"
export SCREEN_MIN_EPOCHS="${SCREEN_MIN_EPOCHS:-2}"
export OT_EPSILONS="${OT_EPSILONS:-0.5,1,2,4}"
export SIGNATURE_MODE="${SIGNATURE_MODE:-family_label_delta_norm}"

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

echo "[delta-mcqa-ot-pca] host=$(hostname) layer=${LAYER} token_position=${TOKEN_POSITION_ID} site_menu=${SITE_MENU} num_bands=${NUM_BANDS} band_scheme=${BAND_SCHEME} top_prefix_sizes=${TOP_PREFIX_SIZES} basis_source_mode=${BASIS_SOURCE_MODE} screen_das=${SCREEN_DAS} ot_eps=${OT_EPSILONS} signature=${SIGNATURE_MODE} timestamp=${RESULTS_TIMESTAMP}"
nvidia-smi

ARGS=(
  --device cuda
  --dataset-path jchang153/copycolors_mcqa
  --layers "${LAYER}"
  --token-position-id "${TOKEN_POSITION_ID}"
  --site-menu "${SITE_MENU}"
  --num-bands "${NUM_BANDS}"
  --band-scheme "${BAND_SCHEME}"
  --top-prefix-sizes "${TOP_PREFIX_SIZES}"
  --basis-source-mode "${BASIS_SOURCE_MODE}"
  --ot-epsilons "${OT_EPSILONS}"
  --signature-mode "${SIGNATURE_MODE}"
  --screen-mask-names "${SCREEN_MASK_NAMES}"
  --screen-max-epochs "${SCREEN_MAX_EPOCHS}"
  --screen-min-epochs "${SCREEN_MIN_EPOCHS}"
  --results-root "${RESULTS_ROOT}"
  --results-timestamp "${RESULTS_TIMESTAMP}"
)

if [[ "${SCREEN_DAS}" == "1" ]]; then
  ARGS+=(--screen-das)
fi

srun --unbuffered python mcqa_ot_pca_focus.py "${ARGS[@]}"
