#!/bin/bash
# run_pam.sh

DATA_DIR="/home/pham156/MoE/FuseMoE_poly/data/PAM"


for seed in 42; do
  for pair1 in "pam 12 acc"; do
    read task num_labels primary_metric <<< "$pair1"
    for pair2 in "4 2 0"; do
      read experts top_k shared_experts <<< "$pair2"
      for router in "joint"; do
        for gating in "poly"; do
          if [ "$gating" = "poly" ]; then
            for normalized in "True"; do
              for power in 2; do
# for seed in 42 0 1 12 123; do
#   for pair1 in "pam 12 f1"; do
#     read task num_labels primary_metric <<< "$pair1"
#     for pair2 in "4 2 0" "3 1 1"; do
#       read experts top_k shared_experts <<< "$pair2"
#       for router in "joint" "permod"; do
#         for gating in "softmax" "laplace" "gaussian" "poly"; do
#           if [ "$gating" = "poly" ]; then
#             for normalized in "True" "False"; do
#               for power in 2 4 6 8; do
                sbatch job_pam.sbatch \
                  --seed "$seed" \
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
            sbatch job_pam.sbatch \
            --seed "$seed" \
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