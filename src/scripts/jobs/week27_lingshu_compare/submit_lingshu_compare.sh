#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/job_lingshu_compare.sbatch"

SEEDS=(${SEEDS:-32 42 52 62 72})
RATIO="${RATIO:-1.0}"
EPOCHS="${EPOCHS:-16}"
ARCHES=(${ARCHES:-lingshu base_shared})

for arch in "${ARCHES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    job_name="w27_${arch}_${seed}_all3"
    echo "Submitting arch=$arch seed=$seed all-3-tasks ratio=$RATIO epochs=$EPOCHS no-ECG"
    sbatch \
      -J "$job_name" \
      --export=ALL,ARCH="$arch",SEED="$seed",RATIO="$RATIO",EPOCHS="$EPOCHS",MODELTYPE=TS_CXR_Text,NUM_MODALITIES=3 \
      "$JOB_SCRIPT"
  done
done
