#!/bin/bash
#SBATCH --job-name=mcqa-full-das
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

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -r "${HOME}/.secrets/hf_token" ]]; then
    export HF_TOKEN="$(< "${HOME}/.secrets/hf_token")"
  else
    echo "[delta-mcqa-full-das] HF_TOKEN is unset and ${HOME}/.secrets/hf_token is not readable"
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
export TARGET_VARS="${TARGET_VARS:-answer_pointer,answer_token}"
export TOKEN_POSITION_IDS="${TOKEN_POSITION_IDS:-last_token}"
export LAYERS="${LAYERS:-auto}"
export DAS_SUBSPACE_DIMS="${DAS_SUBSPACE_DIMS:-32,64,96,128,256,512,768,1024,1536,2048,2304}"
export DAS_MAX_EPOCHS="${DAS_MAX_EPOCHS:-100}"
export DAS_MIN_EPOCHS="${DAS_MIN_EPOCHS:-5}"
export DAS_PLATEAU_PATIENCE="${DAS_PLATEAU_PATIENCE:-2}"
export DAS_PLATEAU_REL_DELTA="${DAS_PLATEAU_REL_DELTA:-0.001}"
export DAS_LEARNING_RATE="${DAS_LEARNING_RATE:-0.001}"
export DAS_RESTARTS="${DAS_RESTARTS:-2}"
export CALIBRATION_METRIC="${CALIBRATION_METRIC:-family_weighted_macro_exact_acc}"
export CALIBRATION_FAMILY_WEIGHTS="${CALIBRATION_FAMILY_WEIGHTS:-1,1.5,2}"

mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

echo "[delta-mcqa-full-das] host=$(hostname) timestamp=${RESULTS_TIMESTAMP}"
echo "[delta-mcqa-full-das] layers=${LAYERS} token_position_ids=${TOKEN_POSITION_IDS}"
echo "[delta-mcqa-full-das] das_subspace_dims=${DAS_SUBSPACE_DIMS}"
echo "[delta-mcqa-full-das] das_restarts=${DAS_RESTARTS}"
echo "[delta-mcqa-full-das] hf_home=${HF_HOME}"
nvidia-smi

srun --unbuffered python mcqa_run_cloud.py \
  --preset full \
  --device cuda \
  --model-name google/gemma-2-2b \
  --dataset-path jchang153/copycolors_mcqa \
  --dataset-size 2000 \
  --split-seed "${SPLIT_SEED:-0}" \
  --train-pool-size "${TRAIN_POOL_SIZE:-200}" \
  --calibration-pool-size "${CALIBRATION_POOL_SIZE:-100}" \
  --test-pool-size "${TEST_POOL_SIZE:-100}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --methods das \
  --target-vars "${TARGET_VARS}" \
  --layers "${LAYERS}" \
  --token-position-ids "${TOKEN_POSITION_IDS}" \
  --resolutions full \
  --calibration-metric "${CALIBRATION_METRIC}" \
  --calibration-family-weights "${CALIBRATION_FAMILY_WEIGHTS}" \
  --das-max-epochs "${DAS_MAX_EPOCHS}" \
  --das-min-epochs "${DAS_MIN_EPOCHS}" \
  --das-plateau-patience "${DAS_PLATEAU_PATIENCE}" \
  --das-plateau-rel-delta "${DAS_PLATEAU_REL_DELTA}" \
  --das-learning-rate "${DAS_LEARNING_RATE}" \
  --das-restarts "${DAS_RESTARTS}" \
  --das-subspace-dims "${DAS_SUBSPACE_DIMS}" \
  --results-root "${RESULTS_ROOT}" \
  --results-timestamp "${RESULTS_TIMESTAMP}_full_das_timed" \
  --signatures-dir signatures
