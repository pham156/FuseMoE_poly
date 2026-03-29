#!/bin/bash
# run_pheno.sh

for seed in 42 0 1 12 123; do
  for pair1 in "ihm-48-cxr-notes-ecg 2 f1" "los-48-cxr-notes-ecg 2 f1"; do
    read task num_labels primary_metric <<< "$pair1"
    for pair2 in "3 1 1"; do
      read experts top_k shared_experts <<< "$pair2"
      for pair3 in "True True" "True False"; do
        read noisy normalized <<< "$pair3"

          
        # Submit one job for this combination
        sbatch job_one_exp.sbatch \
          --seed "$seed" \
          --task "$task" \
          --num_labels "$num_labels" \
          --primary_metric "$primary_metric" \
          --experts "$experts" \
          --top_k "$top_k" \
          --noisy "$noisy" \
          --normalized "$normalized" \
          --shared_experts "$shared_experts"

      done
    done
  done
done

