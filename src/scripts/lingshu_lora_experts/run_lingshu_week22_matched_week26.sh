#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/job_lingshu_week22_matched_week26.sbatch"

# Same seeds used by Week22 reproduce_softmax4_sample_efficiency.
SEEDS=(30 102 20 1 50 22 123 42 0 32)

# Same three MIMIC-IV multimodal tasks.
TASKS=(
  "ihm-48-cxr-notes-ecg 2 f1"
  "los-48-cxr-notes-ecg 2 f1"
  "pheno-all-cxr-notes-ecg 25 macro_f1"
)

for task_spec in "${TASKS[@]}"; do
  read -r task num_labels primary_metric <<< "$task_spec"
  for seed in "${SEEDS[@]}"; do
    echo "Submitting Lingshu Week22-matched task=$task seed=$seed ratios=0.025,0.05,0.1,0.2,0.4,0.6,0.8,1.0"
    sbatch "$JOB_SCRIPT" \
      --task "$task" \
      --num_labels "$num_labels" \
      --primary_metric "$primary_metric" \
      --seed "$seed"
  done
done
