#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "No active Slurm allocation. Allocate an A40 before running this script." >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/u/${USER}/causal-abstractions-ot}"
VENV_PATH="${VENV_PATH:-/work/nvme/bgvo/${USER}/venvs/causal-ot-mcqa}"
RUN_ID="${RUN_ID:-mcqa_pca_mib_seed0_$(date +%Y%m%d_%H%M%S)}"
SPLIT_SEED="${SPLIT_SEED:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-/work/nvme/bgvo/${USER}/mcqa_pca_mib_results}"
CACHE_ROOT="${CACHE_ROOT:-/work/nvme/bgvo/${USER}/hf_cache}"
PRECHECK_STAMP="${PRECHECK_STAMP:-${VENV_PATH}/mcqa_preflight.json}"
PYTHON_MODULE="${PYTHON_MODULE:-python}"
PCA_NUM_BANDS_VALUES="${PCA_NUM_BANDS_VALUES:-1,2,4,8,16,32,64}"

if [[ "${MCQA_PCA_MIB_INSIDE_SRUN:-0}" != "1" ]]; then
  exec srun \
    --ntasks=1 \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    env \
      MCQA_PCA_MIB_INSIDE_SRUN=1 \
      REPO_ROOT="${REPO_ROOT}" \
      VENV_PATH="${VENV_PATH}" \
      RUN_ID="${RUN_ID}" \
      SPLIT_SEED="${SPLIT_SEED}" \
      RESULTS_ROOT="${RESULTS_ROOT}" \
      CACHE_ROOT="${CACHE_ROOT}" \
      PRECHECK_STAMP="${PRECHECK_STAMP}" \
      PYTHON_MODULE="${PYTHON_MODULE}" \
      PCA_NUM_BANDS_VALUES="${PCA_NUM_BANDS_VALUES}" \
      HF_TOKEN="${HF_TOKEN:-}" \
      bash "$0"
fi

if ! command -v module >/dev/null 2>&1; then
  echo "Delta's Lmod command is unavailable inside the srun step." >&2
  exit 2
fi
module load "${PYTHON_MODULE}"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Virtual environment not found at ${VENV_PATH}" >&2
  exit 2
fi
if [[ ! -f "${PRECHECK_STAMP}" ]]; then
  echo "Validated MCQA preflight not found at ${PRECHECK_STAMP}" >&2
  echo "Run: bash experiments/mcqa/slurm/setup_delta_mcqa_env.sh" >&2
  exit 2
fi

source "${VENV_PATH}/bin/activate"
cd "${REPO_ROOT}"

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before starting the sweep.}"
export HF_HOME="${CACHE_ROOT}"
export HF_HUB_CACHE="${CACHE_ROOT}/hub"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="/work/nvme/bgvo/${USER}/xdg_cache"
export PIP_CACHE_DIR="/work/nvme/bgvo/${USER}/pip_cache"
export TMPDIR="/work/nvme/bgvo/${USER}/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PIP_REQUIRE_VIRTUALENV=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
unset PYTHONPATH

mkdir -p \
  "${RESULTS_ROOT}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${TMPDIR}"

python -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
python -m pip check
nvidia-smi

HIER_TIMESTAMP="${RUN_ID}_plot_pca"
HIER_ROOT="${RESULTS_ROOT}/${HIER_TIMESTAMP}_mcqa_hierarchical_sweep"
PLOT_PARTITION_REFERENCE="${HIER_ROOT}/mcqa_plot_layer_pos-last_token_sig-family_label_delta_norm.json"
MIB_RUN_NAME="${RUN_ID}_mib"

echo "[pca-mib] run_id=${RUN_ID} split_seed=${SPLIT_SEED} bands=${PCA_NUM_BANDS_VALUES}"

# PLOT-PCA and PLOT-PCA-DAS share Stage A and the same Stage B PCA sweep. The
# guided DAS stage consumes the pooled-calibration winner selected by PLOT-PCA.
python experiments/mcqa/mcqa_delta_hierarchical_sweep.py \
  --device cuda \
  --model-name google/gemma-2-2b \
  --dataset-path jchang153/copycolors_mcqa \
  --dataset-size 2000 \
  --split-seed "${SPLIT_SEED}" \
  --train-pool-size 200 \
  --calibration-pool-size 200 \
  --test-pool-size 200 \
  --batch-size 64 \
  --results-root "${RESULTS_ROOT}" \
  --results-timestamp "${HIER_TIMESTAMP}" \
  --signatures-dir signatures \
  --stages stage_a_plot_layer,stage_b_plot_pca_support,stage_c_plot_das_pca_support \
  --stage-a-token-position-ids last_token \
  --stage-a-transport-methods uot \
  --stage-a-uot-beta-neurals 0.1,0.3,1,3 \
  --target-vars answer_pointer,answer_token \
  --signature-mode family_label_delta_norm \
  --plot-alignment-method ot \
  --ot-epsilons 0.5,1,2,4 \
  --ot-top-k-values 1,2,4 \
  --ot-lambdas 0.5,1,2,4 \
  --calibration-metric iia_acc \
  --stage-b-top-layers-per-var 1 \
  --stage-b-neighbor-radius 0 \
  --stage-b-max-layers-per-var 1 \
  --pca-site-menus partition \
  --pca-basis-source-modes all_variants \
  --pca-num-bands-values "${PCA_NUM_BANDS_VALUES}" \
  --pca-band-scheme equal \
  --stage-c-top-configs-per-var 1 \
  --guided-max-epochs 100 \
  --guided-min-epochs 5 \
  --screen-restarts 1 \
  --guided-restarts 1

if [[ ! -f "${PLOT_PARTITION_REFERENCE}" ]]; then
  echo "Missing PLOT partition reference at ${PLOT_PARTITION_REFERENCE}" >&2
  exit 1
fi

# Full-vector and DBM canonical/PCA/SAE use exactly the PLOT fit/calibration/test
# partition. Candidate artifacts are resumable and test only the selected layer.
python experiments/mcqa/mcqa_dbm_baselines.py \
  --device cuda \
  --model-name google/gemma-2-2b \
  --dataset-path jchang153/copycolors_mcqa \
  --dataset-size 2000 \
  --methods full-layer,dbm-canonical,dbm-pca,dbm-sae \
  --targets answer_pointer,answer_token \
  --layers all \
  --seeds "${SPLIT_SEED}" \
  --train-size 200 \
  --calibration-size 200 \
  --test-size 200 \
  --batch-size 64 \
  --filter-batch-size 64 \
  --eval-batch-size 128 \
  --partition-reference "${PLOT_PARTITION_REFERENCE}" \
  --epochs 8 \
  --learning-rate 0.01 \
  --temperature-start 1.0 \
  --temperature-end 0.01 \
  --regularization-coefficient 0.0 \
  --sae-release gemma-scope-2b-pt-res-canonical \
  --sae-id-template 'layer_{layer}/width_16k/canonical' \
  --results-root "${RESULTS_ROOT}" \
  --run-name "${MIB_RUN_NAME}" \
  --num-shards 1 \
  --shard-index 0

echo "[pca-mib] complete seed=${SPLIT_SEED}"
echo "[pca-mib] PLOT-PCA outputs: ${HIER_ROOT}"
echo "[pca-mib] DBM/full-vector outputs: ${RESULTS_ROOT}/${MIB_RUN_NAME}"
