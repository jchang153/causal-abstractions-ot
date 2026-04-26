#!/bin/bash
#SBATCH --job-name=mcqa-broad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=48:00:00
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

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before submitting the job.}"
export HF_HOME="${SCRATCH:-/tmp/${USER}}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
export RESULTS_TIMESTAMP="${RESULTS_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export STAGES="${STAGES:-vanilla_ot,pca_ot,pca_guided_das,regular_das}"
export LAYERS="${LAYERS:-20,25}"
export FULL_TOKEN_POSITION_IDS="${FULL_TOKEN_POSITION_IDS:-correct_symbol,correct_symbol_period,last_token}"
export SIGNATURE_MODE="${SIGNATURE_MODE:-family_label_delta_norm}"
export OT_EPSILONS="${OT_EPSILONS:-0.5,1,2,4}"
export OT_TOP_K_VALUES="${OT_TOP_K_VALUES:-1,2,4}"
export OT_LAMBDAS="${OT_LAMBDAS:-0.5,1,2,4}"
export CALIBRATION_METRIC="${CALIBRATION_METRIC:-family_weighted_macro_exact_acc}"
export CALIBRATION_FAMILY_WEIGHTS="${CALIBRATION_FAMILY_WEIGHTS:-1,1.5,2}"
export REGULAR_DAS_SUBSPACE_DIMS="${REGULAR_DAS_SUBSPACE_DIMS:-32,64,96,128,256,512,768,1024,1536,2048,2304}"
export REGULAR_DAS_MAX_EPOCHS="${REGULAR_DAS_MAX_EPOCHS:-100}"
export REGULAR_DAS_MIN_EPOCHS="${REGULAR_DAS_MIN_EPOCHS:-5}"
export REGULAR_DAS_PLATEAU_PATIENCE="${REGULAR_DAS_PLATEAU_PATIENCE:-2}"
export REGULAR_DAS_PLATEAU_REL_DELTA="${REGULAR_DAS_PLATEAU_REL_DELTA:-0.001}"
export REGULAR_DAS_LEARNING_RATE="${REGULAR_DAS_LEARNING_RATE:-0.001}"
export PCA_TOKEN_POSITION_ID="${PCA_TOKEN_POSITION_ID:-last_token}"
export PCA_SITE_MENUS="${PCA_SITE_MENUS:-partition,mixed}"
export PCA_BASIS_SOURCE_MODES="${PCA_BASIS_SOURCE_MODES:-pair_bank,all_variants}"
export PCA_NUM_BANDS="${PCA_NUM_BANDS:-8}"
export PCA_BAND_SCHEME="${PCA_BAND_SCHEME:-equal}"
export PCA_TOP_PREFIX_SIZES="${PCA_TOP_PREFIX_SIZES:-8,16,32,64}"
export GUIDED_PCA_CONFIGS="${GUIDED_PCA_CONFIGS:-pair_bank:partition,all_variants:partition}"
export GUIDED_MASK_NAMES="${GUIDED_MASK_NAMES:-Top1,Top2,Top4,S80}"
export GUIDED_MAX_EPOCHS="${GUIDED_MAX_EPOCHS:-100}"
export GUIDED_MIN_EPOCHS="${GUIDED_MIN_EPOCHS:-5}"
export GUIDED_SUBSPACE_DIMS="${GUIDED_SUBSPACE_DIMS:-}"

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

echo "[delta-mcqa-broad] host=$(hostname) stages=${STAGES} layers=${LAYERS} signature=${SIGNATURE_MODE} timestamp=${RESULTS_TIMESTAMP}"
nvidia-smi

ARGS=(
  --device cuda
  --dataset-path jchang153/copycolors_mcqa
  --layers "${LAYERS}"
  --results-root "${RESULTS_ROOT}"
  --results-timestamp "${RESULTS_TIMESTAMP}"
  --signatures-dir signatures
  --stages "${STAGES}"
  --full-token-position-ids "${FULL_TOKEN_POSITION_IDS}"
  --signature-mode "${SIGNATURE_MODE}"
  --ot-epsilons "${OT_EPSILONS}"
  --ot-top-k-values "${OT_TOP_K_VALUES}"
  --ot-lambdas "${OT_LAMBDAS}"
  --calibration-metric "${CALIBRATION_METRIC}"
  --calibration-family-weights "${CALIBRATION_FAMILY_WEIGHTS}"
  --regular-das-subspace-dims "${REGULAR_DAS_SUBSPACE_DIMS}"
  --regular-das-max-epochs "${REGULAR_DAS_MAX_EPOCHS}"
  --regular-das-min-epochs "${REGULAR_DAS_MIN_EPOCHS}"
  --regular-das-plateau-patience "${REGULAR_DAS_PLATEAU_PATIENCE}"
  --regular-das-plateau-rel-delta "${REGULAR_DAS_PLATEAU_REL_DELTA}"
  --regular-das-learning-rate "${REGULAR_DAS_LEARNING_RATE}"
  --pca-token-position-id "${PCA_TOKEN_POSITION_ID}"
  --pca-site-menus "${PCA_SITE_MENUS}"
  --pca-basis-source-modes "${PCA_BASIS_SOURCE_MODES}"
  --pca-num-bands "${PCA_NUM_BANDS}"
  --pca-band-scheme "${PCA_BAND_SCHEME}"
  --pca-top-prefix-sizes "${PCA_TOP_PREFIX_SIZES}"
  --guided-pca-configs "${GUIDED_PCA_CONFIGS}"
  --guided-mask-names "${GUIDED_MASK_NAMES}"
  --guided-max-epochs "${GUIDED_MAX_EPOCHS}"
  --guided-min-epochs "${GUIDED_MIN_EPOCHS}"
)

if [[ -n "${GUIDED_SUBSPACE_DIMS}" ]]; then
  ARGS+=(--guided-subspace-dims "${GUIDED_SUBSPACE_DIMS}")
fi

srun --unbuffered python mcqa_delta_broad_sweep.py "${ARGS[@]}"
