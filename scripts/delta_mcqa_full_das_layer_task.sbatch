#!/bin/bash
#SBATCH --job-name=mcqa-full-das-layer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --constraint="scratch"
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
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
export RESULTS_TIMESTAMP="${RESULTS_TIMESTAMP:?Set RESULTS_TIMESTAMP for the layer-parallel DAS sweep.}"
export TARGET_VARS="${TARGET_VARS:-answer_pointer,answer_token}"
export TOKEN_POSITION_IDS="${TOKEN_POSITION_IDS:-last_token}"
export LAYER_INDICES="${LAYER_INDICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}"
export DAS_SUBSPACE_DIMS="${DAS_SUBSPACE_DIMS:-32,64,96,128,256,512,768,1024,1536,2048,2304}"
export DAS_MAX_EPOCHS="${DAS_MAX_EPOCHS:-100}"
export DAS_MIN_EPOCHS="${DAS_MIN_EPOCHS:-5}"
export DAS_PLATEAU_PATIENCE="${DAS_PLATEAU_PATIENCE:-2}"
export DAS_PLATEAU_REL_DELTA="${DAS_PLATEAU_REL_DELTA:-0.001}"
export DAS_LEARNING_RATE="${DAS_LEARNING_RATE:-0.001}"
export CALIBRATION_METRIC="${CALIBRATION_METRIC:-family_weighted_macro_exact_acc}"
export CALIBRATION_FAMILY_WEIGHTS="${CALIBRATION_FAMILY_WEIGHTS:-1,1.5,2}"
export SPLIT_SEED="${SPLIT_SEED:-0}"

TASK_INDEX="${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${TASK_INDEX}" ]]; then
  echo "TASK_INDEX or SLURM_ARRAY_TASK_ID is required."
  exit 1
fi

LAYER="$(
  python -c 'import sys; layers=[x.strip() for x in sys.argv[1].split(",") if x.strip()]; print(layers[int(sys.argv[2])])' \
    "${LAYER_INDICES}" "${TASK_INDEX}"
)"

SWEEP_ROOT="${RESULTS_ROOT}/${RESULTS_TIMESTAMP}_full_das_layer_sweep"
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${SWEEP_ROOT}"

echo "[delta-mcqa-full-das-layer] host=$(hostname) timestamp=${RESULTS_TIMESTAMP} split_seed=${SPLIT_SEED}"
echo "[delta-mcqa-full-das-layer] task_index=${TASK_INDEX} layer=${LAYER} token_position_ids=${TOKEN_POSITION_IDS}"
echo "[delta-mcqa-full-das-layer] das_subspace_dims=${DAS_SUBSPACE_DIMS}"
nvidia-smi

srun --unbuffered python mcqa_run_cloud.py \
  --preset full \
  --device cuda \
  --model-name google/gemma-2-2b \
  --dataset-path jchang153/copycolors_mcqa \
  --dataset-size 2000 \
  --split-seed "${SPLIT_SEED}" \
  --train-pool-size "${TRAIN_POOL_SIZE:-200}" \
  --calibration-pool-size "${CALIBRATION_POOL_SIZE:-100}" \
  --test-pool-size "${TEST_POOL_SIZE:-100}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --methods das \
  --target-vars "${TARGET_VARS}" \
  --layers "${LAYER}" \
  --token-position-ids "${TOKEN_POSITION_IDS}" \
  --resolutions full \
  --calibration-metric "${CALIBRATION_METRIC}" \
  --calibration-family-weights "${CALIBRATION_FAMILY_WEIGHTS}" \
  --das-max-epochs "${DAS_MAX_EPOCHS}" \
  --das-min-epochs "${DAS_MIN_EPOCHS}" \
  --das-plateau-patience "${DAS_PLATEAU_PATIENCE}" \
  --das-plateau-rel-delta "${DAS_PLATEAU_REL_DELTA}" \
  --das-learning-rate "${DAS_LEARNING_RATE}" \
  --das-subspace-dims "${DAS_SUBSPACE_DIMS}" \
  --results-root "${SWEEP_ROOT}" \
  --results-timestamp "layer_${LAYER}_full_das_timed" \
  --signatures-dir signatures
