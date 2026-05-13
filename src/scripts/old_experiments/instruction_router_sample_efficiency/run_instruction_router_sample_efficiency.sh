#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/job_instruction_router_sample_efficiency.sbatch"

# Small pilot: enough to compare router behavior without launching the full grid.
SEEDS=(30 20 1)

TASKS=(
  "ihm-48-cxr-notes-ecg 2 f1"
  "los-48-cxr-notes-ecg 2 f1"
  "pheno-all-cxr-notes-ecg 25 macro_f1"
)

# Five configs matching the useful Week 22 space:
# - shared-pool joint/permod with 4 or 3 experts and top-k 2
# - disjoint with 6 experts, split as 2 local experts per modality
CONFIGS=(
  "joint 4 2 2 instr_joint4_k2"
  "joint 3 2 2 instr_joint3_k2"
  "permod 4 2 2 instr_permod4_k2"
  "permod 3 2 2 instr_permod3_k2"
  "disjoint 6 2 2 instr_disjoint6_k2"
)

for task_spec in "${TASKS[@]}"; do
  read -r task num_labels primary_metric <<< "$task_spec"

  for config in "${CONFIGS[@]}"; do
    read -r router experts top_k disjoint_top_k config_name <<< "$config"

    for seed in "${SEEDS[@]}"; do
      echo "Submitting instruction-router pilot task=$task router=$router seed=$seed experts=$experts top_k=$top_k"
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
