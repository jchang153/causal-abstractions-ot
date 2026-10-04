#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python experiments/mcqa/mcqa_run_cloud.py \
  --preset full --device cuda --model-name google/gemma-2-2b \
  --dataset-path jchang153/copycolors_mcqa --dataset-config "" \
  --dataset-size 2000 --split-seed "${SPLIT_SEED:-0}" \
  --training-seed "${TRAINING_SEED:-0}" \
  --train-pool-size 200 --calibration-pool-size 200 --test-pool-size 200 \
  --batch-size 64 --filter-batch-size 64 --methods das \
  --target-vars "${TARGET_VARS:-answer_token}" \
  --pair-bank-target-vars answer_pointer,answer_token \
  --counterfactual-names answerPosition,randomLetter,answerPosition_randomLetter \
  --layers "${LAYERS:-auto}" --token-position-ids last_token --resolutions full \
  --calibration-metric iia_acc --das-max-epochs 100 --das-min-epochs 5 \
  --das-plateau-patience 1 --das-plateau-rel-delta 0.001 \
  --das-learning-rate 0.001 --das-restarts 1 \
  --das-subspace-dims "${DAS_SUBSPACE_DIMS:-128}" \
  --results-root "${RESULTS_ROOT:-results}" \
  --results-timestamp "${RESULTS_TIMESTAMP:-baseline_das_$(date +%Y%m%d_%H%M%S)}" \
  --signatures-dir signatures
