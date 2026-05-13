#!/bin/bash
set -euo pipefail

SCRIPT_ROOT="/home/pham156/MoE/FuseMoE_poly/src/scripts"
OLD_EXPERIMENT_DIR="$SCRIPT_ROOT/old_experiments"
cd "$SCRIPT_ROOT"

for pair1 in "ihm-48-cxr-notes-ecg 2 f1" "los-48-cxr-notes-ecg 2 f1" "pheno-all-cxr-notes-ecg 25 macro_f1"; do
# for pair1 in "pheno-all-cxr-notes-ecg 25 macro_f1"; do
  read -r task num_labels primary_metric <<< "$pair1"
  for router in "joint" "permod"; do
    for pair2 in "4 2 0" "3 1 1"; do
      read -r experts top_k shared_experts <<< "$pair2"

      for gating in "softmax" "gaussian" "laplace"; do
        sbatch "$OLD_EXPERIMENT_DIR/job_week17_full.sbatch" \
          --task "$task" \
          --num_labels "$num_labels" \
          --primary_metric "$primary_metric" \
          --gating "$gating" \
          --experts "$experts" \
          --top_k "$top_k" \
          --shared_experts "$shared_experts" \
          --normalized True \
          --router "$router"
      done

      for pair3 in "poly True" "poly False"; do
        read -r gating normalized <<< "$pair3"
        for power in 2 4 6 8; do
          sbatch "$OLD_EXPERIMENT_DIR/job_week17_full.sbatch" \
            --task "$task" \
            --num_labels "$num_labels" \
            --primary_metric "$primary_metric" \
            --gating "$gating" \
            --experts "$experts" \
            --top_k "$top_k" \
            --shared_experts "$shared_experts" \
            --normalized "$normalized" \
            --router "$router" \
            --poly_power "$power"
        done
      done
    done
  done
done
