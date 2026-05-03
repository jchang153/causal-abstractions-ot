#!/bin/bash
#SBATCH --job-name=mcqa-hpar-a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=06:00:00
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

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before submitting the job.}"
export HF_HOME="${SCRATCH:-/tmp/${USER}}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export RESULTS_ROOT="${RESULTS_ROOT:-results/anvil}"
export RESULTS_TIMESTAMP="${RESULTS_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export STAGE_A_TOKEN_POSITION_IDS="${STAGE_A_TOKEN_POSITION_IDS:-last_token}"
export STAGE_A_LAYER_INDICES="${STAGE_A_LAYER_INDICES:-}"
export TARGET_VARS="${TARGET_VARS:-answer_pointer,answer_token}"
export STAGE_A_METHOD="${STAGE_A_METHOD:-ot}"
export SIGNATURE_MODE="${SIGNATURE_MODE:-family_label_delta_norm}"
export OT_EPSILONS="${OT_EPSILONS:-0.5,1,2,4}"
export UOT_BETA_NEURALS="${UOT_BETA_NEURALS:-0.03,0.1,0.3,1,3}"
export OT_TOP_K_VALUES="${OT_TOP_K_VALUES:-1,2,4}"
export OT_LAMBDAS="${OT_LAMBDAS:-0.5,1,2,4}"
export CALIBRATION_METRIC="${CALIBRATION_METRIC:-family_weighted_macro_exact_acc}"
export CALIBRATION_FAMILY_WEIGHTS="${CALIBRATION_FAMILY_WEIGHTS:-1,1.5,2}"
export SPLIT_SEED="${SPLIT_SEED:-0}"

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

echo "[anvil-mcqa-hpar-stage-a] host=$(hostname) token_positions=${STAGE_A_TOKEN_POSITION_IDS} timestamp=${RESULTS_TIMESTAMP} split_seed=${SPLIT_SEED}"
echo "[anvil-mcqa-hpar-stage-a] stage_a_method=${STAGE_A_METHOD} uot_beta_neurals=${UOT_BETA_NEURALS}"
nvidia-smi

ARGS=(
  --device cuda
  --dataset-path jchang153/copycolors_mcqa
  --results-root "${RESULTS_ROOT}"
  --results-timestamp "${RESULTS_TIMESTAMP}"
  --signatures-dir signatures
  --stage-a-token-position-ids "${STAGE_A_TOKEN_POSITION_IDS}"
  --target-vars "${TARGET_VARS}"
  --stage-a-method "${STAGE_A_METHOD}"
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
