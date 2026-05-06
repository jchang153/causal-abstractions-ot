#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${VENV_PATH:-/root/venvs/caot_fast}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

task_count() {
  python - "$1" <<'PY'
import json, sys
p = sys.argv[1]
with open(p) as f:
    payload = json.load(f)
tasks = payload.get("tasks", payload if isinstance(payload, list) else [])
print(len(tasks))
PY
}

run_task_file() {
  local task_file="$1"
  local label="$2"
  if [[ ! -s "$task_file" ]]; then
    echo "[runpod-hpar] missing/empty task file: $task_file"
    return 0
  fi

  local n
  n="$(task_count "$task_file")"
  echo "[runpod-hpar] ${label}: ${n} tasks from ${task_file}"

  local i
  for ((i=0; i<n; i++)); do
    echo "[runpod-hpar] ${label} task ${i}/${n}"
    python -u mcqa_delta_hierarchical_parallel.py run-task \
      --task-file "$task_file" \
      --task-index "$i"
  done
}
