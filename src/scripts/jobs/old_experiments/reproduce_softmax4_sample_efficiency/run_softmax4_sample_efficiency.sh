#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/job_softmax4_sample_efficiency.sbatch"

# Picked from Week_19 full-data softmax results.
# Seed 112 scored high but had incomplete rows, so it is excluded for cleaner reproduction.
SEEDS=(30 102 20 1 50 22 123 42 0 32)

TASKS=(
  "ihm-48-cxr-notes-ecg 2 f1"
  "los-48-cxr-notes-ecg 2 f1"
  "pheno-all-cxr-notes-ecg 25 macro_f1"
)

# Shared-pool router configs requested: 4 experts/top-k 2 and 3 experts/top-k 2.
SHARED_ROUTER_CONFIGS=(
  "4 2"
  "3 2"
)

# Disjoint splits num_experts across 3 modalities, so 4/2 and 3/2 are not valid disjoint settings.
# These are the clean runnable disjoint baselines.
DISJOINT_CONFIGS=(
  "3 1"
  "6 2"
)

for task_spec in "${TASKS[@]}"; do
  read -r task num_labels primary_metric <<< "$task_spec"

  for router in joint permod; do
    for config in "${SHARED_ROUTER_CONFIGS[@]}"; do
      read -r experts top_k <<< "$config"
      disjoint_top_k="$top_k"
      config_name="softmax_${router}${experts}_k${top_k}"
      for seed in "${SEEDS[@]}"; do
        echo "Submitting task=$task router=$router seed=$seed experts=$experts top_k=$top_k ratios=0.025,0.05,0.1,0.2,0.4,1.0"
        sbatch "$JOB_SCRIPT" \
          --task "$task" \
          --num_labels "$num_labels" \
          --primary_metric "$primary_metric" \
          --seed "$seed" \
          --router "$router" \
          --experts "$experts" \
          --top_k "$top_k" \
          --disjoint_top_k "$disjoint_top_k" \
          --config_name "$config_name"
      done
    done
  done

  for config in "${DISJOINT_CONFIGS[@]}"; do
    read -r experts top_k <<< "$config"
    router="disjoint"
    disjoint_top_k="$top_k"
    config_name="softmax_disjoint${experts}_k${top_k}"
    for seed in "${SEEDS[@]}"; do
      echo "Submitting task=$task router=$router seed=$seed experts=$experts top_k=$top_k ratios=0.025,0.05,0.1,0.2,0.4,1.0"
      sbatch "$JOB_SCRIPT" \
        --task "$task" \
        --num_labels "$num_labels" \
        --primary_metric "$primary_metric" \
        --seed "$seed" \
        --router "$router" \
        --experts "$experts" \
        --top_k "$top_k" \
        --disjoint_top_k "$disjoint_top_k" \
        --config_name "$config_name"
    done
  done
done
