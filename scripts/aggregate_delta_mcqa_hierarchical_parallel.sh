#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:?Usage: bash scripts/aggregate_delta_mcqa_hierarchical_parallel.sh <timestamp>}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"

VENV_PATH="${VENV_PATH:-${HOME}/venvs/causal-ot}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Missing virtual environment at ${VENV_PATH}"
  exit 1
fi
source "${VENV_PATH}/bin/activate"

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
if [[ "${ALLOW_PARTIAL:-0}" == "1" ]]; then
  COMMON_ARGS+=(--allow-partial)
fi

python mcqa_delta_hierarchical_parallel.py aggregate-final "${COMMON_ARGS[@]}"

ROOT="${RESULTS_ROOT}/${TIMESTAMP}_mcqa_hierarchical_sweep"
echo "[aggregate-delta-hpar] summary: ${ROOT}/hierarchical_parallel_summary.txt"
