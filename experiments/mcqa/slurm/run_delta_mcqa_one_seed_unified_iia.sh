#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "No active Slurm allocation. Allocate an A40 first with salloc." >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/u/${USER}/causal-abstractions-ot}"
VENV_PATH="${VENV_PATH:-/work/nvme/bgvo/${USER}/venvs/causal-ot-mcqa}"
RUN_ID="${RUN_ID:-mcqa_unified_iia_seed0_$(date +%Y%m%d_%H%M%S)}"
SPLIT_SEED="${SPLIT_SEED:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-/work/nvme/bgvo/${USER}/mcqa_unified_iia_results}"
CACHE_ROOT="${CACHE_ROOT:-/work/nvme/bgvo/${USER}/hf_cache}"
PRECHECK_STAMP="${PRECHECK_STAMP:-${VENV_PATH}/mcqa_preflight.json}"
PYTHON_MODULE="${PYTHON_MODULE:-python}"

if [[ "${MCQA_ONE_SEED_INSIDE_SRUN:-0}" != "1" ]]; then
  exec srun \
    --ntasks=1 \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    env \
      MCQA_ONE_SEED_INSIDE_SRUN=1 \
      REPO_ROOT="${REPO_ROOT}" \
      VENV_PATH="${VENV_PATH}" \
      RUN_ID="${RUN_ID}" \
      SPLIT_SEED="${SPLIT_SEED}" \
      RESULTS_ROOT="${RESULTS_ROOT}" \
      CACHE_ROOT="${CACHE_ROOT}" \
      PRECHECK_STAMP="${PRECHECK_STAMP}" \
      PYTHON_MODULE="${PYTHON_MODULE}" \
      HF_TOKEN="${HF_TOKEN:-}" \
      bash "$0"
fi

if ! command -v module >/dev/null 2>&1; then
  echo "Delta's Lmod command is unavailable inside the srun step." >&2
  exit 2
fi
module load gcc "${PYTHON_MODULE}"

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

HIER_TIMESTAMP="${RUN_ID}_plot"
HIER_ROOT="${RESULTS_ROOT}/${HIER_TIMESTAMP}_mcqa_hierarchical_sweep"
STAGE_A_RANKINGS="${HIER_ROOT}/stage_a_last_token_layer_rankings.json"
PAPER_RUNTIME="${HIER_ROOT}/paper_runtime_summary.json"
FULL_DAS_TIMESTAMP="${RUN_ID}_full_das"
FULL_DAS_OUTPUT="${RESULTS_ROOT}/${FULL_DAS_TIMESTAMP}_mcqa/mcqa_run_results.json"
BDAS_TIMESTAMP="${RUN_ID}_bdas"
PLOT_BDAS_TIMESTAMP="${RUN_ID}_plot_bdas"
MIB_RUN_NAME="${RUN_ID}_mib"

echo "[one-seed] run_id=${RUN_ID} split_seed=${SPLIT_SEED}"
echo "[one-seed] results_root=${RESULTS_ROOT}"

# One hierarchical invocation computes UOT Stage A exactly once. Stage B native
# and PCA consume those rankings, and their guided DAS stages consume the cached
# Stage B supports. The runtime summarizer adds each shared dependency once.
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
  --stages stage_a_plot_layer,stage_b_plot_native_support,stage_b_plot_pca_support,stage_c_plot_das_native_support,stage_c_plot_das_pca_support \
  --stage-a-token-position-ids last_token \
  --stage-a-transport-methods uot \
  --stage-a-uot-beta-neurals 0.1,0.3,1,3 \
  --target-vars answer_pointer,answer_token \
  --signature-mode family_label_delta_norm \
  --plot-alignment-method ot \
  --ot-epsilons 0.5,1,2,4 \
  --ot-top-k-values 1,2,4 \
  --ot-lambdas 0.5,1,2,4 \
  --calibration-metric family_weighted_macro_iia_acc \
  --calibration-family-weights 1,1,1 \
  --stage-b-top-layers-per-var 1 \
  --stage-b-neighbor-radius 0 \
  --stage-b-max-layers-per-var 1 \
  --native-resolutions 16,32,48,64,128,144,192,256,288,384,576,768 \
  --pca-site-menus partition \
  --pca-basis-source-modes all_variants \
  --pca-num-bands-values 8,16 \
  --pca-band-scheme equal \
  --stage-c-top-configs-per-var 1 \
  --guided-max-epochs 100 \
  --guided-min-epochs 5 \
  --screen-restarts 1 \
  --guided-restarts 2

if [[ ! -f "${STAGE_A_RANKINGS}" || ! -f "${PAPER_RUNTIME}" ]]; then
  echo "Missing Stage A rankings or runtime summary under ${HIER_ROOT}" >&2
  exit 1
fi

AP_LAYER="$(jq -er '._display_method_by_var.answer_pointer.layer // .answer_pointer[0].layer' "${STAGE_A_RANKINGS}")"
AT_LAYER="$(jq -er '._display_method_by_var.answer_token.layer // .answer_token[0].layer' "${STAGE_A_RANKINGS}")"
STAGE_A_SECONDS="$(jq -er '.stage_a_runtime_seconds' "${PAPER_RUNTIME}")"
echo "[one-seed] cached Stage A layers: answer_pointer=${AP_LAYER} answer_token=${AT_LAYER} runtime=${STAGE_A_SECONDS}s"

# Full DAS is independent of PLOT and searches every layer/dimension pair.
if [[ ! -f "${FULL_DAS_OUTPUT}" ]] || ! jq -e \
  '.. | objects | select(.metric_name? == "normalized_full_vocab_top1_v1")' \
  "${FULL_DAS_OUTPUT}" >/dev/null; then
  python experiments/mcqa/mcqa_run_cloud.py \
    --preset full \
    --device cuda \
    --model-name google/gemma-2-2b \
    --dataset-path jchang153/copycolors_mcqa \
    --dataset-size 2000 \
    --split-seed "${SPLIT_SEED}" \
    --train-pool-size 200 \
    --calibration-pool-size 200 \
    --test-pool-size 200 \
    --batch-size 64 \
    --methods das \
    --target-vars answer_pointer,answer_token \
    --layers auto \
    --token-position-ids last_token \
    --resolutions full \
    --calibration-metric family_weighted_macro_iia_acc \
    --calibration-family-weights 1,1,1 \
    --das-max-epochs 100 \
    --das-min-epochs 5 \
    --das-plateau-patience 2 \
    --das-plateau-rel-delta 0.001 \
    --das-learning-rate 0.001 \
    --das-restarts 2 \
    --das-subspace-dims 32,64,96,128,256,512,768,1024,1536,2048,2304 \
    --results-root "${RESULTS_ROOT}" \
    --results-timestamp "${FULL_DAS_TIMESTAMP}" \
    --signatures-dir signatures
else
  echo "[resume] Full DAS ${FULL_DAS_OUTPUT}"
fi

# Full bDAS searches every layer. PLOT-bDAS searches only the one UOT-selected
# layer per variable and reports Stage A once plus its downstream bDAS runtime.
python experiments/mcqa/mcqa_boundless_das.py \
  --device cuda \
  --model-name google/gemma-2-2b \
  --dataset-path jchang153/copycolors_mcqa \
  --dataset-size 2000 \
  --split-seed "${SPLIT_SEED}" \
  --train-pool-size 200 \
  --calibration-pool-size 200 \
  --test-pool-size 200 \
  --batch-size 64 \
  --layers all \
  --token-position-id last_token \
  --target-vars answer_pointer,answer_token \
  --epochs 12 \
  --rotation-learning-rate 0.01 \
  --boundary-learning-rate 0.0001 \
  --boundary-init 0.5 \
  --boundary-penalty 1.0 \
  --temperature-start 1.0 \
  --temperature-end 0.1 \
  --restarts 1 \
  --seed 42 \
  --method-name boundless_das \
  --results-root "${RESULTS_ROOT}" \
  --results-timestamp "${BDAS_TIMESTAMP}" \
  --signatures-dir signatures \
  --resume

python experiments/mcqa/mcqa_boundless_das.py \
  --device cuda \
  --model-name google/gemma-2-2b \
  --dataset-path jchang153/copycolors_mcqa \
  --dataset-size 2000 \
  --split-seed "${SPLIT_SEED}" \
  --train-pool-size 200 \
  --calibration-pool-size 200 \
  --test-pool-size 200 \
  --batch-size 64 \
  --layers-by-target "answer_pointer:${AP_LAYER},answer_token:${AT_LAYER}" \
  --token-position-id last_token \
  --target-vars answer_pointer,answer_token \
  --epochs 12 \
  --rotation-learning-rate 0.01 \
  --boundary-learning-rate 0.0001 \
  --boundary-init 0.5 \
  --boundary-penalty 1.0 \
  --temperature-start 1.0 \
  --temperature-end 0.1 \
  --restarts 1 \
  --seed 42 \
  --method-name plot_bdas \
  --upstream-runtime-seconds "${STAGE_A_SECONDS}" \
  --results-root "${RESULTS_ROOT}" \
  --results-timestamp "${PLOT_BDAS_TIMESTAMP}" \
  --signatures-dir signatures \
  --resume

# Full-layer and all three DBM variants share one model/data load and resume at
# individual layer artifacts. Test is evaluated only for each selected layer.
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
  --eval-batch-size 128 \
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

# Refresh the hierarchical table-facing runtime summary with Full DAS included.
python experiments/mcqa/mcqa_paper_runtime.py \
  "${HIER_ROOT}" \
  --full-das-output "${FULL_DAS_OUTPUT}"

echo "[one-seed] complete: ${RESULTS_ROOT}"
echo "[one-seed] PLOT runtimes: ${HIER_ROOT}/paper_runtime_summary.txt"
echo "[one-seed] Full DAS: ${FULL_DAS_OUTPUT}"
echo "[one-seed] Full bDAS: ${RESULTS_ROOT}/${BDAS_TIMESTAMP}_mcqa_boundless_das"
echo "[one-seed] PLOT-bDAS: ${RESULTS_ROOT}/${PLOT_BDAS_TIMESTAMP}_mcqa_boundless_das"
echo "[one-seed] MIB baselines: ${RESULTS_ROOT}/${MIB_RUN_NAME}"
