#!/bin/bash
# Avoid set -u because Conda hooks can read unset env vars.
set -eo pipefail

LINGSHU_MODEL_PATH="${LINGSHU_MODEL_PATH:-/scratch/gilbreth/pham156/hf_cache/lingshu-medical-mllm/Lingshu-7B}"

if command -v module >/dev/null 2>&1; then
  module load anaconda
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
fi

echo "Python: $(command -v python)"
python - <<'PY'
import importlib.util
for pkg in ["torch", "transformers", "accelerate", "peft"]:
    print(f"{pkg}: {bool(importlib.util.find_spec(pkg))}")
PY

if [[ -d "$LINGSHU_MODEL_PATH" && -z "$(find "$LINGSHU_MODEL_PATH" -maxdepth 1 -type f -name config.json -print -quit)" ]]; then
  SNAPSHOT="$(find "$LINGSHU_MODEL_PATH" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  if [[ -n "$SNAPSHOT" ]]; then
    LINGSHU_MODEL_PATH="$SNAPSHOT"
  fi
fi

echo "LINGSHU_MODEL_PATH=$LINGSHU_MODEL_PATH"
if [[ ! -f "$LINGSHU_MODEL_PATH/config.json" ]]; then
  echo "Missing Lingshu config.json at $LINGSHU_MODEL_PATH" >&2
  echo "Set LINGSHU_MODEL_PATH to a local Lingshu snapshot before submitting Lingshu jobs." >&2
  exit 1
fi

echo "Found Lingshu config:"
ls -lh "$LINGSHU_MODEL_PATH/config.json"
