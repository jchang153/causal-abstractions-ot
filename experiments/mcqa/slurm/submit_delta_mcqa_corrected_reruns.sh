#!/bin/bash

set -euo pipefail

TIMESTAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/delta}"
DELTA_ACCOUNT="${DELTA_ACCOUNT:-bgvo-delta-gpu}"
DELTA_PARTITION="${DELTA_PARTITION:-gpuA40x4}"

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before submitting the corrected MCQA reruns.}"

PLOT_ALL_STAGES="stage_a_plot_layer,stage_b_plot_native_support,stage_b_plot_pca_support,stage_c_plot_das_layer,stage_c_plot_das_dimension,stage_c_plot_das_pca_support"
PLOT_ONLY_STAGES="stage_a_plot_layer,stage_b_plot_native_support,stage_b_plot_pca_support"

echo "[submit-mcqa-corrected] timestamp=${TIMESTAMP}"
echo "[submit-mcqa-corrected] results_root=${RESULTS_ROOT}"

for split_seed in 0 1 2; do
  for alignment_method in ot cosine bruteforce; do
    stages="${PLOT_ONLY_STAGES}"
    if [[ "${alignment_method}" == "ot" ]]; then
      stages="${PLOT_ALL_STAGES}"
    fi
    run_timestamp="${TIMESTAMP}_${alignment_method}_seed${split_seed}"
    echo "[submit-mcqa-corrected] PLOT seed=${split_seed} alignment=${alignment_method}"
    RESULTS_ROOT="${RESULTS_ROOT}" \
    RESULTS_TIMESTAMP="${run_timestamp}" \
    SPLIT_SEED="${split_seed}" \
    PLOT_ALIGNMENT_METHOD="${alignment_method}" \
    STAGES="${stages}" \
    DATASET_SIZE=2000 \
    TRAIN_POOL_SIZE=200 \
    CALIBRATION_POOL_SIZE=200 \
    TEST_POOL_SIZE=200 \
    sbatch \
      --account="${DELTA_ACCOUNT}" \
      --partition="${DELTA_PARTITION}" \
      --export=ALL \
      --job-name="mcqa-${alignment_method}-s${split_seed}" \
      experiments/mcqa/slurm/delta_mcqa_hierarchical_sweep.sbatch
  done
done

for split_seed in 0 1 2; do
  echo "[submit-mcqa-corrected] Full DAS seed=${split_seed}"
  RESULTS_ROOT="${RESULTS_ROOT}" \
  RESULTS_TIMESTAMP="${TIMESTAMP}_full_das_seed${split_seed}" \
  SPLIT_SEED="${split_seed}" \
  TRAIN_POOL_SIZE=200 \
  CALIBRATION_POOL_SIZE=200 \
  TEST_POOL_SIZE=200 \
  sbatch \
    --account="${DELTA_ACCOUNT}" \
    --partition="${DELTA_PARTITION}" \
    --export=ALL \
    --job-name="mcqa-full-das-s${split_seed}" \
    experiments/mcqa/slurm/delta_mcqa_full_das_timed.sbatch
done

echo "[submit-mcqa-corrected] bDAS seeds=0,1,2"
RESULTS_ROOT="${RESULTS_ROOT}" \
RESULTS_TIMESTAMP="${TIMESTAMP}_bdas" \
sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --export=ALL \
  experiments/mcqa/slurm/delta_mcqa_boundless_das.sbatch

echo "[submit-mcqa-corrected] MIB baselines seeds=0,1,2"
REPO_ROOT="${REPO_ROOT:-${PWD}}" \
RESULTS_ROOT="${RESULTS_ROOT}" \
RUN_NAME="${TIMESTAMP}_mcqa_mib_corrected" \
sbatch \
  --account="${DELTA_ACCOUNT}" \
  --partition="${DELTA_PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --gpus-per-node=1 \
  --mem=64g \
  --time=2-00:00:00 \
  --export=ALL \
  --job-name=mcqa-mib-corrected \
  experiments/mcqa/slurm/run_delta_mcqa_mib_baselines.sh

echo "[submit-mcqa-corrected] all jobs submitted"
