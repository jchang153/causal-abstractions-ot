#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:?Usage: bash scripts/submit_delta_mcqa_hierarchical_parallel_stage_c.sh <timestamp>}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"

DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-4}"
SPLIT_SEED="${SPLIT_SEED:-0}"

VENV_PATH="${VENV_PATH:-${HOME}/venvs/causal-ot}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing virtual environment at ${VENV_PATH}"
  exit 1
fi
source "${VENV_PATH}/bin/activate"

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before planning/submitting Stage C.}"

echo "[submit-delta-hpar-c] timestamp=${TIMESTAMP}"
echo "[submit-delta-hpar-c] delta_account=${DELTA_ACCOUNT}"
echo "[submit-delta-hpar-c] delta_partition=${DELTA_PARTITION}"
echo "[submit-delta-hpar-c] array_throttle=${ARRAY_THROTTLE}"
echo "[submit-delta-hpar-c] regular_das_subspace_dims=${REGULAR_DAS_SUBSPACE_DIMS:-32,64,96,128,256,512,768,1024,1536,2048,2304}"
echo "[submit-delta-hpar-c] guided_subspace_dims=${GUIDED_SUBSPACE_DIMS:-32,64,96,128,256,512,768,1024,1536,2048,2304}"
echo "[submit-delta-hpar-c] split_seed=${SPLIT_SEED}"

COMMON_ARGS=(
  --device cuda
  --dataset-path jchang153/copycolors_mcqa
  --results-root "${RESULTS_ROOT}"
  --results-timestamp "${TIMESTAMP}"
  --signatures-dir signatures
  --stage-a-token-position-ids "${STAGE_A_TOKEN_POSITION_IDS:-last_token}"
  --target-vars "${TARGET_VARS:-answer_pointer,answer_token}"
  --signature-mode "${SIGNATURE_MODE:-family_label_delta_norm}"
  --ot-epsilons "${OT_EPSILONS:-0.5,1,2,4}"
  --ot-top-k-values "${OT_TOP_K_VALUES:-1,2,4}"
  --ot-lambdas "${OT_LAMBDAS:-0.5,1,2,4}"
  --calibration-metric "${CALIBRATION_METRIC:-family_weighted_macro_exact_acc}"
  --calibration-family-weights "${CALIBRATION_FAMILY_WEIGHTS:-1,1.5,2}"
  --split-seed "${SPLIT_SEED}"
  --stage-b-top-layers-per-var "${STAGE_B_TOP_LAYERS_PER_VAR:-1}"
  --stage-b-neighbor-radius "${STAGE_B_NEIGHBOR_RADIUS:-0}"
  --stage-b-max-layers-per-var "${STAGE_B_MAX_LAYERS_PER_VAR:-1}"
  --native-block-resolutions "${NATIVE_BLOCK_RESOLUTIONS:-128,144,192,256,288,384,576,768}"
  --pca-site-menus "${PCA_SITE_MENUS:-partition,mixed}"
  --pca-basis-source-modes "${PCA_BASIS_SOURCE_MODES:-pair_bank,all_variants}"
  --pca-num-bands-values "${PCA_NUM_BANDS_VALUES:-8,16}"
  --pca-band-scheme "${PCA_BAND_SCHEME:-equal}"
  --pca-top-prefix-sizes "${PCA_TOP_PREFIX_SIZES:-8,16,32,64}"
  --stage-c-top-configs-per-var "${STAGE_C_TOP_CONFIGS_PER_VAR:-2}"
  --guided-mask-names "${GUIDED_MASK_NAMES:-Top1,Top2,Top4,S50,S80}"
  --guided-max-epochs "${GUIDED_MAX_EPOCHS:-100}"
  --guided-min-epochs "${GUIDED_MIN_EPOCHS:-5}"
  --screen-restarts "${SCREEN_RESTARTS:-1}"
  --guided-restarts "${GUIDED_RESTARTS:-2}"
  --regular-das-subspace-dims "${REGULAR_DAS_SUBSPACE_DIMS:-32,64,96,128,256,512,768,1024,1536,2048,2304}"
)

if [[ -n "${STAGE_A_LAYER_INDICES:-}" ]]; then
  COMMON_ARGS+=(--stage-a-layer-indices "${STAGE_A_LAYER_INDICES}")
fi
if [[ -n "${GUIDED_SUBSPACE_DIMS:-}" ]]; then
  COMMON_ARGS+=(--guided-subspace-dims "${GUIDED_SUBSPACE_DIMS}")
fi

python mcqa_delta_hierarchical_parallel.py aggregate-stage-b --skip-native-aggregation "${COMMON_ARGS[@]}"
python mcqa_delta_hierarchical_parallel.py plan-stage-c "${COMMON_ARGS[@]}"

ROOT="${RESULTS_ROOT}/${TIMESTAMP}_mcqa_hierarchical_sweep"
STAGE_C_TASK_FILE="${ROOT}/stage_c_pca_tasks.json"
STAGE_C_TASKS="$(
  python - "${STAGE_C_TASK_FILE}" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text())
print(len(payload.get("tasks", [])))
PY
)"

if [[ "${STAGE_C_TASKS}" -le 0 ]]; then
  echo "[submit-delta-hpar-c] no Stage C tasks"
  exit 0
fi

ARRAY_SPEC="0-$((STAGE_C_TASKS - 1))%${ARRAY_THROTTLE}"
echo "[submit-delta-hpar-c] submitting Stage C tasks=${STAGE_C_TASKS} array=${ARRAY_SPEC}"
sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --array="${ARRAY_SPEC}" \
  --export=ALL,TASK_FILE="${STAGE_C_TASK_FILE}" \
  --job-name=mcqa-hpar-c \
  scripts/delta_mcqa_hierarchical_parallel_task.sbatch

echo "[submit-delta-hpar-c] task plan: ${ROOT}/stage_c_task_plan.txt"
