#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"

DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"

echo "[submit-delta-broad] timestamp=${TIMESTAMP}"
echo "[submit-delta-broad] delta_account=${DELTA_ACCOUNT}"
echo "[submit-delta-broad] delta_partition=${DELTA_PARTITION}"
echo "[submit-delta-broad] stages=${STAGES:-vanilla_ot,pca_ot,pca_guided_das,regular_das}"
echo "[submit-delta-broad] layers=${LAYERS:-20,25}"
echo "[submit-delta-broad] full_token_position_ids=${FULL_TOKEN_POSITION_IDS:-correct_symbol,correct_symbol_period,last_token}"
echo "[submit-delta-broad] signature_mode=${SIGNATURE_MODE:-family_label_delta_norm}"
echo "[submit-delta-broad] pca_site_menus=${PCA_SITE_MENUS:-partition}"
echo "[submit-delta-broad] pca_basis_source_modes=${PCA_BASIS_SOURCE_MODES:-all_variants}"
echo "[submit-delta-broad] guided_pca_configs=${GUIDED_PCA_CONFIGS:-all_variants:partition}"
echo "[submit-delta-broad] guided_mask_names=${GUIDED_MASK_NAMES:-Selected}"
echo "[submit-delta-broad] regular_das_subspace_dims=${REGULAR_DAS_SUBSPACE_DIMS:-32,64,96,128,256,512,768,1024,1536,2048,2304}"

RESULTS_TIMESTAMP="${TIMESTAMP}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --export=ALL \
  --job-name=mcqa-broad \
  scripts/delta_mcqa_broad_sweep.sbatch

echo "[submit-delta-broad] results root will be ${RESULTS_ROOT}/${TIMESTAMP}_mcqa_broad_sweep"
