#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/anvil}"
LAYERS=("${@:2}")

if [[ ${#LAYERS[@]} -eq 0 ]]; then
  LAYERS=(20 25)
fi

echo "[submit-ot-pca] timestamp=${TIMESTAMP}"
echo "[submit-ot-pca] layers=${LAYERS[*]}"
echo "[submit-ot-pca] token_position_id=${TOKEN_POSITION_ID:-last_token}"
echo "[submit-ot-pca] num_bands=${NUM_BANDS:-8}"
echo "[submit-ot-pca] band_scheme=${BAND_SCHEME:-equal}"
echo "[submit-ot-pca] basis_source_mode=${BASIS_SOURCE_MODE:-all_variants}"
echo "[submit-ot-pca] ot_epsilons=${OT_EPSILONS:-0.5,1,2,4}"
echo "[submit-ot-pca] signature_mode=${SIGNATURE_MODE:-family_label_delta_norm}"
echo "[submit-ot-pca] exclude_nodes=${EXCLUDE_NODES:-none}"

SBATCH_ARGS=(--export=ALL)
if [[ -n "${EXCLUDE_NODES:-}" ]]; then
  SBATCH_ARGS+=(--exclude="${EXCLUDE_NODES}")
fi

for LAYER in "${LAYERS[@]}"; do
  echo "[submit-ot-pca] submitting layer=${LAYER}"
  LAYER="${LAYER}" \
  RESULTS_TIMESTAMP="${TIMESTAMP}" \
  RESULTS_ROOT="${RESULTS_ROOT}" \
  TOKEN_POSITION_ID="${TOKEN_POSITION_ID:-last_token}" \
  NUM_BANDS="${NUM_BANDS:-8}" \
  BAND_SCHEME="${BAND_SCHEME:-equal}" \
  BASIS_SOURCE_MODE="${BASIS_SOURCE_MODE:-all_variants}" \
  OT_EPSILONS="${OT_EPSILONS:-0.5,1,2,4}" \
  SIGNATURE_MODE="${SIGNATURE_MODE:-family_label_delta_norm}" \
  sbatch "${SBATCH_ARGS[@]}" scripts/anvil_mcqa_ot_pca_single.sbatch
done

echo "[submit-ot-pca] results root will be ${RESULTS_ROOT}/${TIMESTAMP}_mcqa_ot_pca_focus"
