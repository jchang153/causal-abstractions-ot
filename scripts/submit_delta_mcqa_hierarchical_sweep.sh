#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"

DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA100x4}"

echo "[submit-delta-hier] timestamp=${TIMESTAMP}"
echo "[submit-delta-hier] delta_account=${DELTA_ACCOUNT}"
echo "[submit-delta-hier] delta_partition=${DELTA_PARTITION}"
echo "[submit-delta-hier] stages=${STAGES:-stage_a_layer_ot,stage_b_native_ot,stage_b_pca_ot,stage_c_guided_das}"
echo "[submit-delta-hier] stage_a_token_position_ids=${STAGE_A_TOKEN_POSITION_IDS:-last_token}"
echo "[submit-delta-hier] stage_b_top_layers_per_var=${STAGE_B_TOP_LAYERS_PER_VAR:-1}"
echo "[submit-delta-hier] stage_b_neighbor_radius=${STAGE_B_NEIGHBOR_RADIUS:-0}"
echo "[submit-delta-hier] stage_b_max_layers_per_var=${STAGE_B_MAX_LAYERS_PER_VAR:-1}"
echo "[submit-delta-hier] native_resolutions=${NATIVE_RESOLUTIONS:-}"
echo "[submit-delta-hier] pca_site_menus=${PCA_SITE_MENUS:-partition}"
echo "[submit-delta-hier] pca_basis_source_modes=${PCA_BASIS_SOURCE_MODES:-pair_bank,all_variants}"
echo "[submit-delta-hier] pca_num_bands_values=${PCA_NUM_BANDS_VALUES:-8,16}"
echo "[submit-delta-hier] stage_c_top_configs_per_var=${STAGE_C_TOP_CONFIGS_PER_VAR:-2}"
echo "[submit-delta-hier] guided_mask_names=${GUIDED_MASK_NAMES:-Selected}"
echo "[submit-delta-hier] screen_restarts=${SCREEN_RESTARTS:-1}"
echo "[submit-delta-hier] guided_restarts=${GUIDED_RESTARTS:-2}"
echo "[submit-delta-hier] regular_das_subspace_dims=${REGULAR_DAS_SUBSPACE_DIMS:-32,64,96,128,256,512,768,1024,1536,2048,2304}"
echo "[submit-delta-hier] guided_subspace_dims=${GUIDED_SUBSPACE_DIMS:-auto}"

RESULTS_TIMESTAMP="${TIMESTAMP}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --export=ALL \
  --job-name=mcqa-hier \
  scripts/delta_mcqa_hierarchical_sweep.sbatch

echo "[submit-delta-hier] results root will be ${RESULTS_ROOT}/${TIMESTAMP}_mcqa_hierarchical_sweep"
