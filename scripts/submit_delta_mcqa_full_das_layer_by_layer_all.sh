#!/bin/bash

set -euo pipefail

RUN_TAG="${1:-$(date +%Y%m%d_mcqa_full_das_layer_by_layer)}"
SEEDS="${SEEDS:-0,1,2}"
RESULTS_BASE="${RESULTS_BASE:-results/delta_full_das_layer_by_layer}"
LAYER_INDICES="${LAYER_INDICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-4}"
FULL_DAS_LAYER_TIME="${FULL_DAS_LAYER_TIME:-02:00:00}"
DAS_RESTARTS="${DAS_RESTARTS:-2}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -r "${HOME}/.secrets/hf_token" ]]; then
    export HF_TOKEN="$(< "${HOME}/.secrets/hf_token")"
  else
    echo "[submit-delta-full-das-layer-by-layer] HF_TOKEN is unset and ${HOME}/.secrets/hf_token is not readable"
    exit 1
  fi
fi

IFS=',' read -r -a SEED_ITEMS <<< "${SEEDS}"
JOB_IDS=()

for raw_seed in "${SEED_ITEMS[@]}"; do
  SEED="$(echo "${raw_seed}" | xargs)"
  if [[ -z "${SEED}" ]]; then
    continue
  fi

  TS="${RUN_TAG}_seed${SEED}"
  ROOT="${RESULTS_BASE}_seed${SEED}"

  echo "===== submit full-DAS layer-by-layer seed=${SEED} ====="
  echo "[submit-delta-full-das-layer-by-layer] timestamp=${TS}"
  echo "[submit-delta-full-das-layer-by-layer] results_root=${ROOT}"
  echo "[submit-delta-full-das-layer-by-layer] layers=${LAYER_INDICES}"
  echo "[submit-delta-full-das-layer-by-layer] array_throttle=${ARRAY_THROTTLE}"
  echo "[submit-delta-full-das-layer-by-layer] time_limit=${FULL_DAS_LAYER_TIME}"

  SUBMIT_OUTPUT="$({
    RESULTS_ROOT="${ROOT}" \
    SPLIT_SEED="${SEED}" \
    LAYER_INDICES="${LAYER_INDICES}" \
    ARRAY_THROTTLE="${ARRAY_THROTTLE}" \
    FULL_DAS_LAYER_TIME="${FULL_DAS_LAYER_TIME}" \
    DAS_RESTARTS="${DAS_RESTARTS}" \
    bash scripts/submit_delta_mcqa_full_das_layer_parallel.sh "${TS}"
  } 2>&1)"
  echo "${SUBMIT_OUTPUT}"

  JOB_ID="$(echo "${SUBMIT_OUTPUT}" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
  if [[ -n "${JOB_ID}" ]]; then
    JOB_IDS+=("${JOB_ID}")
  else
    echo "[submit-delta-full-das-layer-by-layer] warning: could not parse job id for seed=${SEED}"
  fi

  echo "[submit-delta-full-das-layer-by-layer] aggregate after completion:"
  echo "  python mcqa_full_das_layer_aggregate.py ${ROOT}/${TS}_full_das_layer_sweep --expected-layers ${LAYER_INDICES}"
done

if [[ "${#JOB_IDS[@]}" -gt 0 ]]; then
  IDS_CSV="$(IFS=,; echo "${JOB_IDS[*]}")"
  echo "===== submitted full-DAS layer-by-layer arrays ====="
  echo "job_ids=${IDS_CSV}"
  echo "monitor with:"
  echo "  squeue -j ${IDS_CSV} -o \"%.18i %.28j %.8T %.10M %.10l %.30R %.20S\""
  echo "  sacct -j ${IDS_CSV} --parsable2 --noheader --format=JobID,JobName,State,ExitCode,Elapsed"
fi
