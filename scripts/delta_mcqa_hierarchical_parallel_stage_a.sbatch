#!/bin/bash
#SBATCH --job-name=mcqa-hpar-a
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

cd "${SLURM_SUBMIT_DIR}"

VENV_PATH="${VENV_PATH:-${HOME}/venvs/causal-ot}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing virtual environment at ${VENV_PATH}"
  exit 1
fi
source "${VENV_PATH}/bin/activate"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -r "${HOME}/.secrets/hf_token" ]]; then
    export HF_TOKEN="$(< "${HOME}/.secrets/hf_token")"
  else
    echo "[delta-mcqa-hpar-stage-a] HF_TOKEN is unset and ${HOME}/.secrets/hf_token is not readable"
    exit 1
  fi
fi
export HF_HOME="${HF_HOME:-${SCRATCH:-/tmp/${USER}}/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
export RESULTS_TIMESTAMP="${RESULTS_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export STAGE_A_TOKEN_POSITION_IDS="${STAGE_A_TOKEN_POSITION_IDS:-last_token}"
export STAGE_A_LAYER_INDICES="${STAGE_A_LAYER_INDICES:-}"
export TARGET_VARS="${TARGET_VARS:-answer_pointer,answer_token}"
export STAGE_A_METHOD="${STAGE_A_METHOD:-uot}"
export STAGE_A_HPARAM_SELECTION="${STAGE_A_HPARAM_SELECTION:-rowwise}"
export STAGE_A_RERANK_TOP_K="${STAGE_A_RERANK_TOP_K:-0}"
export STAGE_A_RERANK_DROP_RATIO="${STAGE_A_RERANK_DROP_RATIO:-0}"
export STAGE_A_RERANK_MIN_K="${STAGE_A_RERANK_MIN_K:-6}"
export STAGE_A_RERANK_MAX_K="${STAGE_A_RERANK_MAX_K:-8}"
export SIGNATURE_MODE="${SIGNATURE_MODE:-family_label_delta_norm}"
export OT_EPSILONS="${OT_EPSILONS:-0.5,1,2,4}"
export UOT_BETA_NEURALS="${UOT_BETA_NEURALS:-0.03,0.1,0.3,1,3}"
export OT_TOP_K_VALUES="${OT_TOP_K_VALUES:-1,2,4}"
export OT_LAMBDAS="${OT_LAMBDAS:-0.5,1,2,4}"
export CALIBRATION_METRIC="${CALIBRATION_METRIC:-family_weighted_macro_exact_acc}"
export CALIBRATION_FAMILY_WEIGHTS="${CALIBRATION_FAMILY_WEIGHTS:-1,1.5,2}"
export SPLIT_SEED="${SPLIT_SEED:-0}"
export BATCH_SIZE="${BATCH_SIZE:-64}"

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

echo "[delta-mcqa-hpar-stage-a] host=$(hostname) token_positions=${STAGE_A_TOKEN_POSITION_IDS} timestamp=${RESULTS_TIMESTAMP} split_seed=${SPLIT_SEED}"
echo "[delta-mcqa-hpar-stage-a] stage_a_method=${STAGE_A_METHOD} stage_a_hparam_selection=${STAGE_A_HPARAM_SELECTION} stage_a_rerank_top_k=${STAGE_A_RERANK_TOP_K} stage_a_rerank_drop_ratio=${STAGE_A_RERANK_DROP_RATIO} stage_a_rerank_min_k=${STAGE_A_RERANK_MIN_K} stage_a_rerank_max_k=${STAGE_A_RERANK_MAX_K} uot_beta_neurals=${UOT_BETA_NEURALS} batch_size=${BATCH_SIZE}"
echo "[delta-mcqa-hpar-stage-a] hf_home=${HF_HOME}"
nvidia-smi

ARGS=(
  --device cuda
  --dataset-path jchang153/copycolors_mcqa
  --batch-size "${BATCH_SIZE}"
  --results-root "${RESULTS_ROOT}"
  --results-timestamp "${RESULTS_TIMESTAMP}"
  --signatures-dir signatures
  --stage-a-token-position-ids "${STAGE_A_TOKEN_POSITION_IDS}"
  --target-vars "${TARGET_VARS}"
  --stage-a-method "${STAGE_A_METHOD}"
  --stage-a-hparam-selection "${STAGE_A_HPARAM_SELECTION}"
  --stage-a-rerank-top-k "${STAGE_A_RERANK_TOP_K}"
  --stage-a-rerank-drop-ratio "${STAGE_A_RERANK_DROP_RATIO}"
  --stage-a-rerank-min-k "${STAGE_A_RERANK_MIN_K}"
  --stage-a-rerank-max-k "${STAGE_A_RERANK_MAX_K}"
  --signature-mode "${SIGNATURE_MODE}"
  --ot-epsilons "${OT_EPSILONS}"
  --uot-beta-neurals "${UOT_BETA_NEURALS}"
  --ot-top-k-values "${OT_TOP_K_VALUES}"
  --ot-lambdas "${OT_LAMBDAS}"
  --calibration-metric "${CALIBRATION_METRIC}"
  --calibration-family-weights "${CALIBRATION_FAMILY_WEIGHTS}"
  --split-seed "${SPLIT_SEED}"
)

if [[ -n "${STAGE_A_LAYER_INDICES}" ]]; then
  ARGS+=(--stage-a-layer-indices "${STAGE_A_LAYER_INDICES}")
fi

srun --unbuffered python mcqa_delta_hierarchical_parallel.py stage-a "${ARGS[@]}"
