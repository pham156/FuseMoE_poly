#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/job_lingshu_compat_week26.sbatch"

# Mechanical architecture-compatibility check only.
# This is not intended as a performance experiment.
SEED=30
RATIO=0.025
EPOCHS=1

TASKS=(
  "ihm-48-cxr-notes-ecg 2 f1"
  "los-48-cxr-notes-ecg 2 f1"
  "pheno-all-cxr-notes-ecg 25 macro_f1"
)

for task_spec in "${TASKS[@]}"; do
  read -r task num_labels primary_metric <<< "$task_spec"
  echo "Submitting Lingshu compatibility smoke task=$task seed=$SEED ratio=$RATIO"
  sbatch "$JOB_SCRIPT" \
    --task "$task" \
    --num_labels "$num_labels" \
    --primary_metric "$primary_metric" \
    --seed "$SEED" \
    --ratio "$RATIO" \
    --epochs "$EPOCHS"
done
