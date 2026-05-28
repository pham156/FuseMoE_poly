#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/job_semantic_profiles_week25.sbatch"

# Mentor-aligned semantic profile comparison:
# - no instruction router
# - semantic expert profile router enabled
# - 5 seeds
# - 3 router types
# - all 3 MIMIC-IV tasks
SEEDS=(30 20 1 50 42)

TASKS=(
  "ihm-48-cxr-notes-ecg 2 f1"
  "los-48-cxr-notes-ecg 2 f1"
  "pheno-all-cxr-notes-ecg 25 macro_f1"
)

# joint/permod use 4 experts to match the 4 semantic profiles.
# disjoint splits experts across 3 modalities, so 12 total experts gives
# each modality a local 4-profile expert set.
CONFIGS=(
  "joint 4 2 2 sem_profile_joint4_k2"
  "permod 4 2 2 sem_profile_permod4_k2"
  "disjoint 12 2 2 sem_profile_disjoint12_k2"
)

# Five seeds kept from earlier sample-efficiency runs so results are comparable
# with prior Week 23/24 experiments while keeping the first semantic-profile grid
# smaller than the previous 10-seed reproduction.
RATIOS="0.025 0.05 0.1 0.2 0.4 0.6 0.8 1.0"
EPOCHS=8

for task_spec in "${TASKS[@]}"; do
  read -r task num_labels primary_metric <<< "$task_spec"

  for config in "${CONFIGS[@]}"; do
    read -r router experts top_k disjoint_top_k config_name <<< "$config"

    for seed in "${SEEDS[@]}"; do
      echo "Submitting Week25 semantic profile task=$task router=$router seed=$seed experts=$experts top_k=$top_k ratios=$RATIOS"
      sbatch "$JOB_SCRIPT" \
        --task "$task" \
        --num_labels "$num_labels" \
        --primary_metric "$primary_metric" \
        --seed "$seed" \
        --router "$router" \
        --experts "$experts" \
        --top_k "$top_k" \
        --disjoint_top_k "$disjoint_top_k" \
        --config_name "$config_name" \
        --ratios "$RATIOS" \
        --epochs "$EPOCHS"
    done
  done
done
