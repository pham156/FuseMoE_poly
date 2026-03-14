#!/bin/bash
# run_recreate.sh

for seed in 42 0 1 12 123; do
  for pair1 in "ihm-48-cxr-notes-ecg 2 f1" "los-48-cxr-notes-ecg 2 f1" "pheno-all-cxr-notes-ecg 25 macro_f1"; do
    read task num_labels primary_metric <<< "$pair1"
    for pair2 in "3 2"; do
      read experts top_k <<< "$pair2"

      sbatch job_recreate.sbatch \
        --seed "$seed" \
        --task "$task" \
        --num_labels "$num_labels" \
        --primary_metric "$primary_metric" \
        --experts "$experts" \
        --top_k "$top_k"
    done
    
  done
done