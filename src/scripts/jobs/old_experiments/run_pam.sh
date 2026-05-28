#!/bin/bash
SCRIPT_ROOT="/home/pham156/MoE/FuseMoE_poly/src/scripts"
OLD_EXPERIMENT_DIR="$SCRIPT_ROOT/old_experiments"
cd "$SCRIPT_ROOT"

# run_pam.sh

DATA_DIR="/home/pham156/MoE/FuseMoE_poly/data/PAM"
SEED_BATCHES=(
  "batch1:0,1,12,22,32"
  # "batch2:42,52,62,72,82"
  # "batch3:92,102,112,122,123"
)


# for pair1 in "pam 11 acc"; do
#   read task num_labels primary_metric <<< "$pair1"
#   for seed_batch in "${SEED_BATCHES[@]}"; do
#     IFS=':' read -r seed_batch_name seed_list <<< "$seed_batch"
#     for pair2 in "4 2 0" "3 1 1"; do
#       read experts top_k shared_experts <<< "$pair2"
#       for router in "joint" "permod"; do
#         for gating in "softmax" "laplace" "gaussian" "poly"; do
for pair1 in "pam 11 acc"; do
  read task num_labels primary_metric <<< "$pair1"
  for seed_batch in "${SEED_BATCHES[@]}"; do
    IFS=':' read -r seed_batch_name seed_list <<< "$seed_batch"
    for pair2 in "4 2 0"; do
      read experts top_k shared_experts <<< "$pair2"
      for router in "joint"; do
        for gating in "softmax"; do
          if [ "$gating" = "poly" ]; then
            for normalized in "True" "False"; do
              for power in 2 4 6 8; do
                sbatch "$OLD_EXPERIMENT_DIR/job_pam.sbatch" \
                  --seed_batch "$seed_batch_name" \
                  --seed_list "$seed_list" \
                  --task "$task" \
                  --num_labels "$num_labels" \
                  --primary_metric "$primary_metric" \
                  --experts "$experts" \
                  --top_k "$top_k" \
                  --normalized "$normalized" \
                  --shared_experts "$shared_experts" \
                  --router "$router" \
                  --gating "$gating" \
                  --poly_power "$power" \
                  --data_dir "$DATA_DIR"
              done
            done
          else
            sbatch "$OLD_EXPERIMENT_DIR/job_pam.sbatch" \
              --seed_batch "$seed_batch_name" \
              --seed_list "$seed_list" \
              --task "$task" \
              --num_labels "$num_labels" \
              --primary_metric "$primary_metric" \
              --experts "$experts" \
              --top_k "$top_k" \
              --normalized True \
              --shared_experts "$shared_experts" \
              --router "$router" \
              --gating "$gating" \
              --data_dir "$DATA_DIR"
          fi
        done
      done
    done
  done
done
