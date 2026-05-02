#!/bin/bash
#SBATCH --job-name=mcqa-hier
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
export STAGES="${STAGES:-stage_a_layer_ot,stage_b_native_ot,stage_b_pca_ot,stage_c_guided_das}"
export STAGE_A_TOKEN_POSITION_IDS="${STAGE_A_TOKEN_POSITION_IDS:-last_token}"
export STAGE_A_LAYER_INDICES="${STAGE_A_LAYER_INDICES:-}"
export TARGET_VARS="${TARGET_VARS:-answer_pointer,answer_token}"
export SIGNATURE_MODE="${SIGNATURE_MODE:-family_label_delta_norm}"
export OT_EPSILONS="${OT_EPSILONS:-0.5,1,2,4}"
export OT_TOP_K_VALUES="${OT_TOP_K_VALUES:-1,2,4}"
export OT_LAMBDAS="${OT_LAMBDAS:-0.5,1,2,4}"
export CALIBRATION_METRIC="${CALIBRATION_METRIC:-family_weighted_macro_exact_acc}"
export CALIBRATION_FAMILY_WEIGHTS="${CALIBRATION_FAMILY_WEIGHTS:-1,1,1}"
export STAGE_B_TOP_LAYERS_PER_VAR="${STAGE_B_TOP_LAYERS_PER_VAR:-1}"
export STAGE_B_NEIGHBOR_RADIUS="${STAGE_B_NEIGHBOR_RADIUS:-0}"
export STAGE_B_MAX_LAYERS_PER_VAR="${STAGE_B_MAX_LAYERS_PER_VAR:-1}"
export NATIVE_BLOCK_RESOLUTIONS="${NATIVE_BLOCK_RESOLUTIONS:-128,144,192,256,288,384,576,768}"
export PCA_SITE_MENUS="${PCA_SITE_MENUS:-partition,mixed}"
export PCA_BASIS_SOURCE_MODES="${PCA_BASIS_SOURCE_MODES:-pair_bank,all_variants}"
export PCA_NUM_BANDS_VALUES="${PCA_NUM_BANDS_VALUES:-8,16}"
export PCA_BAND_SCHEME="${PCA_BAND_SCHEME:-equal}"
export PCA_TOP_PREFIX_SIZES="${PCA_TOP_PREFIX_SIZES:-8,16,32,64}"
export STAGE_C_TOP_CONFIGS_PER_VAR="${STAGE_C_TOP_CONFIGS_PER_VAR:-2}"
export GUIDED_MASK_NAMES="${GUIDED_MASK_NAMES:-Top1,Top2,Top4,S50,S80}"
export GUIDED_MAX_EPOCHS="${GUIDED_MAX_EPOCHS:-100}"
export GUIDED_MIN_EPOCHS="${GUIDED_MIN_EPOCHS:-5}"
export SCREEN_RESTARTS="${SCREEN_RESTARTS:-1}"
export GUIDED_RESTARTS="${GUIDED_RESTARTS:-2}"
export GUIDED_SUBSPACE_DIMS="${GUIDED_SUBSPACE_DIMS:-32,64,96,128,256,512,768,1024,1536,2048,2304}"
export REGULAR_DAS_SUBSPACE_DIMS="${REGULAR_DAS_SUBSPACE_DIMS:-32,64,96,128,256,512,768,1024,1536,2048,2304}"

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

echo "[delta-mcqa-hier] host=$(hostname) stages=${STAGES} token_positions=${STAGE_A_TOKEN_POSITION_IDS} timestamp=${RESULTS_TIMESTAMP}"
nvidia-smi

ARGS=(
  --device cuda
  --dataset-path jchang153/copycolors_mcqa
  --results-root "${RESULTS_ROOT}"
  --results-timestamp "${RESULTS_TIMESTAMP}"
  --signatures-dir signatures
  --stages "${STAGES}"
  --stage-a-token-position-ids "${STAGE_A_TOKEN_POSITION_IDS}"
  --target-vars "${TARGET_VARS}"
  --signature-mode "${SIGNATURE_MODE}"
  --ot-epsilons "${OT_EPSILONS}"
  --ot-top-k-values "${OT_TOP_K_VALUES}"
  --ot-lambdas "${OT_LAMBDAS}"
  --calibration-metric "${CALIBRATION_METRIC}"
  --calibration-family-weights "${CALIBRATION_FAMILY_WEIGHTS}"
  --stage-b-top-layers-per-var "${STAGE_B_TOP_LAYERS_PER_VAR}"
  --stage-b-neighbor-radius "${STAGE_B_NEIGHBOR_RADIUS}"
  --stage-b-max-layers-per-var "${STAGE_B_MAX_LAYERS_PER_VAR}"
  --native-block-resolutions "${NATIVE_BLOCK_RESOLUTIONS}"
  --pca-site-menus "${PCA_SITE_MENUS}"
  --pca-basis-source-modes "${PCA_BASIS_SOURCE_MODES}"
  --pca-num-bands-values "${PCA_NUM_BANDS_VALUES}"
  --pca-band-scheme "${PCA_BAND_SCHEME}"
  --pca-top-prefix-sizes "${PCA_TOP_PREFIX_SIZES}"
  --stage-c-top-configs-per-var "${STAGE_C_TOP_CONFIGS_PER_VAR}"
  --guided-mask-names "${GUIDED_MASK_NAMES}"
  --guided-max-epochs "${GUIDED_MAX_EPOCHS}"
  --guided-min-epochs "${GUIDED_MIN_EPOCHS}"
  --screen-restarts "${SCREEN_RESTARTS}"
  --guided-restarts "${GUIDED_RESTARTS}"
  --regular-das-subspace-dims "${REGULAR_DAS_SUBSPACE_DIMS}"
)

if [[ -n "${STAGE_A_LAYER_INDICES}" ]]; then
  ARGS+=(--stage-a-layer-indices "${STAGE_A_LAYER_INDICES}")
fi

if [[ -n "${GUIDED_SUBSPACE_DIMS}" ]]; then
  ARGS+=(--guided-subspace-dims "${GUIDED_SUBSPACE_DIMS}")
fi

if [[ -n "${FULL_DAS_OUTPUTS:-}" ]]; then
  IFS=',' read -r -a FULL_DAS_OUTPUT_ARRAY <<< "${FULL_DAS_OUTPUTS}"
  for full_das_output in "${FULL_DAS_OUTPUT_ARRAY[@]}"; do
    if [[ -n "${full_das_output}" ]]; then
      ARGS+=(--full-das-output "${full_das_output}")
    fi
  done
fi

srun --unbuffered python mcqa_delta_hierarchical_sweep.py "${ARGS[@]}"
