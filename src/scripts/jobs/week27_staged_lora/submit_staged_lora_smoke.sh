#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="$SCRIPT_DIR/job_staged_lora_compare.sbatch"

SEED="${SEED:-32}"
TASK="${TASK:-ihm-48-cxr-notes-ecg}"
NUM_LABELS="${NUM_LABELS:-2}"
PRIMARY_METRIC="${PRIMARY_METRIC:-f1}"

for router in joint permod; do
  for arch in base_shared staged_lora; do
    sbatch \
      --job-name "w27_${arch}_${router}" \
      --export=ALL,ARCH="$arch",ROUTER="$router",SEED="$SEED",TASK="$TASK",NUM_LABELS="$NUM_LABELS",PRIMARY_METRIC="$PRIMARY_METRIC",EPOCHS=16,WARMUP_EPOCHS=8 \
      "$JOB"
  done
done
