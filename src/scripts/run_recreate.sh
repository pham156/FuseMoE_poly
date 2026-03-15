#!/bin/bash
# run_recreate.sh

for seed in 42 0 1 12 123; do
  for pair1 in "pheno-all-cxr-notes-ecg 25 macro_f1"; do
    read task num_labels primary_metric <<< "$pair1"
    for pair2 in "4 2 1"; do
      read experts top_k shared_experts <<< "$pair2"

      sbatch job_recreate.sbatch \
        --seed "$seed" \
        --task "$task" \
        --num_labels "$num_labels" \
        --primary_metric "$primary_metric" \
        --experts "$experts" \
        --top_k "$top_k" \
        --shared_experts "$shared_experts"
    done
    
  done
done