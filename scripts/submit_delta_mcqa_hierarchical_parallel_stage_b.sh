#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:?Usage: bash scripts/submit_delta_mcqa_hierarchical_parallel_stage_b.sh <timestamp>}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"

DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-8}"
SUBMIT_NATIVE="${SUBMIT_NATIVE:-1}"
SUBMIT_PCA="${SUBMIT_PCA:-1}"

VENV_PATH="${VENV_PATH:-${HOME}/venvs/causal-ot}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing virtual environment at ${VENV_PATH}"
  exit 1
fi
source "${VENV_PATH}/bin/activate"

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before planning/submitting Stage B.}"

echo "[submit-delta-hpar-b] timestamp=${TIMESTAMP}"
echo "[submit-delta-hpar-b] delta_account=${DELTA_ACCOUNT}"
echo "[submit-delta-hpar-b] delta_partition=${DELTA_PARTITION}"
echo "[submit-delta-hpar-b] array_throttle=${ARRAY_THROTTLE}"
echo "[submit-delta-hpar-b] native_block_resolutions=${NATIVE_BLOCK_RESOLUTIONS:-576,288}"
echo "[submit-delta-hpar-b] pca_site_menus=${PCA_SITE_MENUS:-partition,mixed}"
echo "[submit-delta-hpar-b] pca_basis_source_modes=${PCA_BASIS_SOURCE_MODES:-pair_bank,all_variants}"
echo "[submit-delta-hpar-b] pca_num_bands_values=${PCA_NUM_BANDS_VALUES:-8,16}"

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
  --stage-b-top-layers-per-var "${STAGE_B_TOP_LAYERS_PER_VAR:-3}"
  --stage-b-neighbor-radius "${STAGE_B_NEIGHBOR_RADIUS:-1}"
  --stage-b-max-layers-per-var "${STAGE_B_MAX_LAYERS_PER_VAR:-5}"
  --native-block-resolutions "${NATIVE_BLOCK_RESOLUTIONS:-576,288}"
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
)

if [[ -n "${STAGE_A_LAYER_INDICES:-}" ]]; then
  COMMON_ARGS+=(--stage-a-layer-indices "${STAGE_A_LAYER_INDICES}")
fi
if [[ -n "${GUIDED_SUBSPACE_DIMS:-}" ]]; then
  COMMON_ARGS+=(--guided-subspace-dims "${GUIDED_SUBSPACE_DIMS}")
fi

python mcqa_delta_hierarchical_parallel.py plan-stage-b "${COMMON_ARGS[@]}"

ROOT="${RESULTS_ROOT}/${TIMESTAMP}_mcqa_hierarchical_sweep"
NATIVE_TASK_FILE="${ROOT}/stage_b_native_tasks.json"
PCA_TASK_FILE="${ROOT}/stage_b_pca_tasks.json"

task_count() {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text())
print(len(payload.get("tasks", [])))
PY
}

submit_array() {
  local task_file="$1"
  local task_count="$2"
  local job_name="$3"
  if [[ "${task_count}" -le 0 ]]; then
    echo "[submit-delta-hpar-b] no tasks for ${job_name}"
    return
  fi
  local array_spec="0-$((task_count - 1))%${ARRAY_THROTTLE}"
  echo "[submit-delta-hpar-b] submitting ${job_name}: tasks=${task_count} array=${array_spec} task_file=${task_file}"
  sbatch \
    --account="${DELTA_ACCOUNT}" \
    --partition="${DELTA_PARTITION}" \
    --array="${array_spec}" \
    --export=ALL,TASK_FILE="${task_file}" \
    --job-name="${job_name}" \
    scripts/delta_mcqa_hierarchical_parallel_task.sbatch
}

NATIVE_TASKS="$(task_count "${NATIVE_TASK_FILE}")"
PCA_TASKS="$(task_count "${PCA_TASK_FILE}")"

if [[ "${SUBMIT_NATIVE}" == "1" ]]; then
  submit_array "${NATIVE_TASK_FILE}" "${NATIVE_TASKS}" "mcqa-hpar-native"
fi
if [[ "${SUBMIT_PCA}" == "1" ]]; then
  submit_array "${PCA_TASK_FILE}" "${PCA_TASKS}" "mcqa-hpar-pca"
fi

echo "[submit-delta-hpar-b] task plan: ${ROOT}/stage_b_task_plan.txt"
