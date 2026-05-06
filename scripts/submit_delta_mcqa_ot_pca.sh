#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
LAYERS=("${@:2}")

if [[ ${#LAYERS[@]} -eq 0 ]]; then
  LAYERS=(20 25)
fi

DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"

echo "[submit-delta-ot-pca] timestamp=${TIMESTAMP}"
echo "[submit-delta-ot-pca] layers=${LAYERS[*]}"
echo "[submit-delta-ot-pca] delta_account=${DELTA_ACCOUNT}"
echo "[submit-delta-ot-pca] delta_partition=${DELTA_PARTITION}"
echo "[submit-delta-ot-pca] token_position_id=${TOKEN_POSITION_ID:-last_token}"
echo "[submit-delta-ot-pca] site_menu=${SITE_MENU:-partition}"
echo "[submit-delta-ot-pca] num_bands=${NUM_BANDS:-8}"
echo "[submit-delta-ot-pca] band_scheme=${BAND_SCHEME:-equal}"
echo "[submit-delta-ot-pca] basis_source_mode=${BASIS_SOURCE_MODE:-all_variants}"
echo "[submit-delta-ot-pca] screen_das=${SCREEN_DAS:-0}"
echo "[submit-delta-ot-pca] screen_mask_names=${SCREEN_MASK_NAMES:-Selected}"
echo "[submit-delta-ot-pca] ot_epsilons=${OT_EPSILONS:-0.5,1,2,4}"
echo "[submit-delta-ot-pca] signature_mode=${SIGNATURE_MODE:-family_label_delta_norm}"

SBATCH_ARGS=(
  --account="${DELTA_ACCOUNT}"
  --partition="${DELTA_PARTITION}"
  --export=ALL
)

for LAYER in "${LAYERS[@]}"; do
  echo "[submit-delta-ot-pca] submitting layer=${LAYER}"
  LAYER="${LAYER}" \
  RESULTS_TIMESTAMP="${TIMESTAMP}" \
  RESULTS_ROOT="${RESULTS_ROOT}" \
  TOKEN_POSITION_ID="${TOKEN_POSITION_ID:-last_token}" \
  SITE_MENU="${SITE_MENU:-partition}" \
  NUM_BANDS="${NUM_BANDS:-8}" \
  BAND_SCHEME="${BAND_SCHEME:-equal}" \
  BASIS_SOURCE_MODE="${BASIS_SOURCE_MODE:-all_variants}" \
  SCREEN_DAS="${SCREEN_DAS:-0}" \
  SCREEN_MASK_NAMES="${SCREEN_MASK_NAMES:-Selected}" \
  SCREEN_MAX_EPOCHS="${SCREEN_MAX_EPOCHS:-25}" \
  SCREEN_MIN_EPOCHS="${SCREEN_MIN_EPOCHS:-2}" \
  OT_EPSILONS="${OT_EPSILONS:-0.5,1,2,4}" \
  SIGNATURE_MODE="${SIGNATURE_MODE:-family_label_delta_norm}" \
  sbatch "${SBATCH_ARGS[@]}" scripts/delta_mcqa_ot_pca_single.sbatch
done

echo "[submit-delta-ot-pca] results root will be ${RESULTS_ROOT}/${TIMESTAMP}_mcqa_ot_pca_focus"
