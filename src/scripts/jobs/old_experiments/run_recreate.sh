#!/bin/bash
SCRIPT_ROOT="/home/pham156/MoE/FuseMoE_poly/src/scripts"
OLD_EXPERIMENT_DIR="$SCRIPT_ROOT/old_experiments"
cd "$SCRIPT_ROOT"

# run_recreate.sh


for pair1 in "ihm-48-cxr-notes-ecg 2 f1" "los-48-cxr-notes-ecg 2 f1" "pheno-all-cxr-notes-ecg 25 macro_f1"; do
  read task num_labels primary_metric <<< "$pair1"
  for pair3 in "sigmoid False"; do
    read gating use_temp <<< "$pair3"
    for seed in 0 1 12 22 32 42 52 62 72 82 92 102 112 122 123; do
      for router in "joint"; do
        for pair2 in "4 2 0"; do
          read experts top_k shared_experts <<< "$pair2"

          sbatch "$OLD_EXPERIMENT_DIR/job_recreate.sbatch" \
            --task "$task" \
            --num_labels "$num_labels" \
            --primary_metric "$primary_metric" \
            --seed "$seed" \
            --gating "$gating" \
            --experts "$experts" \
            --top_k "$top_k" \
            --shared_experts "$shared_experts" \
            --normalized "True" \
            --router "$router" \
            --poly_power "2" \
            --use_bias "False" \
            --use_temp "$use_temp"
        done
      done
    done
  done
done
