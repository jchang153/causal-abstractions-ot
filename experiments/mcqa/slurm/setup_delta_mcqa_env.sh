#!/usr/bin/env bash

# Build and fully validate a fresh MCQA environment inside an active Delta GPU allocation.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "No active Slurm allocation. Run salloc first." >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN to a token with access to Gemma and the MCQA dataset." >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/u/${USER}/causal-abstractions-ot}"
NVME_ROOT="${NVME_ROOT:-/work/nvme/bgvo/${USER}}"
VENV_PATH="${VENV_PATH:-${NVME_ROOT}/venvs/causal-ot-mcqa}"
CACHE_ROOT="${CACHE_ROOT:-${NVME_ROOT}/hf_cache}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${NVME_ROOT}/pip_cache}"
TMPDIR="${TMPDIR:-${NVME_ROOT}/tmp}"
PRECHECK_STAMP="${PRECHECK_STAMP:-${VENV_PATH}/mcqa_preflight.json}"
PYTHON_MODULE="${PYTHON_MODULE:-python}"

for path in "${VENV_PATH}" "${CACHE_ROOT}" "${PIP_CACHE_DIR}" "${TMPDIR}" "${PRECHECK_STAMP}"; do
  case "${path}" in
    "${NVME_ROOT}"/*) ;;
    *) echo "Refusing non-NVMe environment/cache path: ${path}" >&2; exit 2 ;;
  esac
done
if [[ ! -f "${REPO_ROOT}/requirements.txt" ]]; then
  echo "Repository not found at ${REPO_ROOT}" >&2
  exit 2
fi

if [[ "${MCQA_SETUP_INSIDE_SRUN:-0}" != "1" ]]; then
  exec srun \
    --ntasks=1 \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    env \
      MCQA_SETUP_INSIDE_SRUN=1 \
      REPO_ROOT="${REPO_ROOT}" \
      NVME_ROOT="${NVME_ROOT}" \
      VENV_PATH="${VENV_PATH}" \
      CACHE_ROOT="${CACHE_ROOT}" \
      PIP_CACHE_DIR="${PIP_CACHE_DIR}" \
      TMPDIR="${TMPDIR}" \
      PRECHECK_STAMP="${PRECHECK_STAMP}" \
      PYTHON_MODULE="${PYTHON_MODULE}" \
      HF_TOKEN="${HF_TOKEN:-}" \
      HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
      bash "$0"
fi

mkdir -p \
  "${NVME_ROOT}/venvs" \
  "${CACHE_ROOT}/hub" \
  "${CACHE_ROOT}/datasets" \
  "${CACHE_ROOT}/transformers" \
  "${PIP_CACHE_DIR}" \
  "${TMPDIR}"

export HF_HOME="${CACHE_ROOT}"
export HF_HUB_CACHE="${CACHE_ROOT}/hub"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${NVME_ROOT}/xdg_cache"
export PIP_CACHE_DIR TMPDIR
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PIP_REQUIRE_VIRTUALENV=true
unset PYTHONPATH

# Delta's OS python3 is currently 3.9.  Load the supported Spack Python before
# creating or activating the venv; the unversioned module tracks Delta's current
# supported Python stack and can be overridden with PYTHON_MODULE if necessary.
if ! command -v module >/dev/null 2>&1; then
  echo "Delta's Lmod command is unavailable inside the srun step." >&2
  exit 2
fi
module load gcc "${PYTHON_MODULE}"
module list

cd "${REPO_ROOT}"
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  PYTHON_BOOTSTRAP=""
  for candidate in python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BOOTSTRAP="$(command -v "${candidate}")"
      break
    fi
  done
  if [[ -z "${PYTHON_BOOTSTRAP}" ]]; then
    echo "No Python 3 interpreter found. Inspect available Delta Python modules with: module spider python" >&2
    exit 2
  fi
  "${PYTHON_BOOTSTRAP}" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), sys.version'
  "${PYTHON_BOOTSTRAP}" -m venv "${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"
python -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), sys.version'
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade --upgrade-strategy only-if-needed -r requirements.txt
python -m pip check
python -m compileall -q experiments/mcqa
python -m pytest -q \
  experiments/mcqa/test_unified_iia.py \
  experiments/mcqa/test_shared_epsilon_selection.py \
  experiments/mcqa/test_boundless_das.py

command -v jq >/dev/null || { echo "jq is required by the sweep launcher but is not on PATH" >&2; exit 2; }
nvidia-smi
python experiments/mcqa/mcqa_environment_preflight.py \
  --cache-root "${CACHE_ROOT}" \
  --stamp "${PRECHECK_STAMP}" \
  --prefetch-all-saes

echo "Environment and artifact preflight passed."
echo "VENV_PATH=${VENV_PATH}"
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "PRECHECK_STAMP=${PRECHECK_STAMP}"
du -sh "${VENV_PATH}" "${CACHE_ROOT}" "${PIP_CACHE_DIR}"
df -h "${NVME_ROOT}"
