#!/bin/bash

set -euo pipefail

BASE_ROOT="${BASE_ROOT:-${PWD}/results/delta_stageA_adaptive_double_10seed}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
TIMESTAMP_PREFIX="${TIMESTAMP_PREFIX:-$(date +%Y%m%d)_mcqa_stageA_adaptive}"
VARIANTS="${VARIANTS:-ot_uot ot_only}"

DELTA_STAGE_A_TIME="${DELTA_STAGE_A_TIME:-00:30:00}"
STAGE_A_HPARAM_SELECTION="${STAGE_A_HPARAM_SELECTION:-rowwise}"
STAGE_A_RERANK_TOP_K="${STAGE_A_RERANK_TOP_K:-0}"
STAGE_A_RERANK_DROP_RATIO="${STAGE_A_RERANK_DROP_RATIO:-0.75}"
STAGE_A_RERANK_MIN_K="${STAGE_A_RERANK_MIN_K:-6}"
STAGE_A_RERANK_MAX_K="${STAGE_A_RERANK_MAX_K:-8}"
UOT_BETA_NEURALS="${UOT_BETA_NEURALS:-0.03,0.3,1,2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT_SLEEP_SECONDS="${SUBMIT_SLEEP_SECONDS:-1}"
STAGE_A_JOB_NAME="${STAGE_A_JOB_NAME:-mcqa-hpar-a}"
SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS:-}"
SKIP_COMPILE="${SKIP_COMPILE:-0}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -r "${HOME}/.secrets/hf_token" ]]; then
    export HF_TOKEN="$(< "${HOME}/.secrets/hf_token")"
  else
    echo "[submit-delta-stage-a-adaptive-double] HF_TOKEN is unset and ${HOME}/.secrets/hf_token is not readable"
    exit 1
  fi
fi

if [[ "${SKIP_COMPILE}" != "1" ]]; then
  python -m py_compile mcqa_delta_hierarchical_sweep.py mcqa_delta_hierarchical_parallel.py mcqa_stage_a_uot_modes_summary.py
fi

mkdir -p "${BASE_ROOT}"
MANIFEST="${BASE_ROOT}/adaptive_double_submissions_$(date +%Y%m%d_%H%M%S).tsv"
echo -e "variant\tseed\tjob_id\tstate\tresults_root\ttimestamp" > "${MANIFEST}"

method_for_variant() {
  case "$1" in
    ot_uot) echo "ot,uot" ;;
    ot_only) echo "ot" ;;
    *)
      echo "[submit-delta-stage-a-adaptive-double] unsupported variant: $1" >&2
      return 1
      ;;
  esac
}

for variant in ${VARIANTS}; do
  method="$(method_for_variant "${variant}")"
  variant_root="${BASE_ROOT}/${variant}"
  mkdir -p "${variant_root}"

  for seed in ${SEEDS}; do
    timestamp="${TIMESTAMP_PREFIX}_${variant}_seed${seed}"
    sweep_root="${variant_root}/${timestamp}_mcqa_hierarchical_sweep"
    ranking_path="${sweep_root}/stage_a_last_token_layer_rankings.json"

    if [[ "${SKIP_EXISTING}" == "1" && -s "${ranking_path}" ]]; then
      echo "[submit-delta-stage-a-adaptive-double] skip existing variant=${variant} seed=${seed} root=${sweep_root}"
      echo -e "${variant}\t${seed}\tSKIPPED\texisting\t${variant_root}\t${timestamp}" >> "${MANIFEST}"
      continue
    fi

    echo "[submit-delta-stage-a-adaptive-double] submitting variant=${variant} method=${method} seed=${seed} timestamp=${timestamp}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo -e "${variant}\t${seed}\tDRY_RUN\tdry_run\t${variant_root}\t${timestamp}" >> "${MANIFEST}"
      continue
    fi

    set +e
    submit_output="$(
      RESULTS_ROOT="${variant_root}" \
      SPLIT_SEED="${seed}" \
      STAGE_A_METHOD="${method}" \
      STAGE_A_HPARAM_SELECTION="${STAGE_A_HPARAM_SELECTION}" \
      STAGE_A_RERANK_TOP_K="${STAGE_A_RERANK_TOP_K}" \
      STAGE_A_RERANK_DROP_RATIO="${STAGE_A_RERANK_DROP_RATIO}" \
      STAGE_A_RERANK_MIN_K="${STAGE_A_RERANK_MIN_K}" \
      STAGE_A_RERANK_MAX_K="${STAGE_A_RERANK_MAX_K}" \
      UOT_BETA_NEURALS="${UOT_BETA_NEURALS}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      DELTA_STAGE_A_TIME="${DELTA_STAGE_A_TIME}" \
      STAGE_A_JOB_NAME="${STAGE_A_JOB_NAME}" \
      SBATCH_EXTRA_ARGS="${SBATCH_EXTRA_ARGS}" \
      bash scripts/submit_delta_mcqa_hierarchical_parallel_stage_a.sh "${timestamp}" 2>&1
    )"
    rc=$?
    set -e
    echo "${submit_output}"
    if [[ "${rc}" -ne 0 ]]; then
      echo -e "${variant}\t${seed}\tERROR\tsubmit_failed\t${variant_root}\t${timestamp}" >> "${MANIFEST}"
      exit "${rc}"
    fi
    job_id="$(grep -Eo 'Submitted batch job [0-9]+' <<< "${submit_output}" | awk '{print $4}' | tail -n 1)"
    job_id="${job_id:-UNKNOWN}"
    echo -e "${variant}\t${seed}\t${job_id}\tsubmitted\t${variant_root}\t${timestamp}" >> "${MANIFEST}"
    sleep "${SUBMIT_SLEEP_SECONDS}"
  done
done

echo "[submit-delta-stage-a-adaptive-double] manifest=${MANIFEST}"
