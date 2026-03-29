#!/bin/bash
# run_recreate.sh


# for pair1 in "ihm-48-cxr-notes-ecg 2 f1" "los-48-cxr-notes-ecg 2 f1" "pheno-all-cxr-notes-ecg 25 macro_f1"; do
#   read task num_labels primary_metric <<< "$pair1"
#   for seed in 42 0 1 12 123; do
#     for pair3 in "softmax False" "sigmoid False" "softmax True" "sigmoid True" "sigmoid_dist True"; do
for pair1 in "los-48-cxr-notes-ecg 2 f1" "pheno-all-cxr-notes-ecg 25 macro_f1"; do
  read task num_labels primary_metric <<< "$pair1"
  for pair3 in "softmax False" "sigmoid False" "softmax True" "sigmoid True" "sigmoid_dist True"; do
    read gating use_temp <<< "$pair3"
    for seed in 42 0 1 12 123; do
      for pair2 in "4 2 0"; do
        read experts top_k shared_experts <<< "$pair2"

        sbatch job_recreate.sbatch \
          --task "$task" \
          --num_labels "$num_labels" \
          --primary_metric "$primary_metric" \
          --seed "$seed" \
          --gating "$gating" \
          --experts "$experts" \
          --top_k "$top_k" \
          --shared_experts "$shared_experts" \
          --use_temp "$use_temp"
      done
    done
  done
done
