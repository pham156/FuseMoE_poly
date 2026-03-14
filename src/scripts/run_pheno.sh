#!/bin/bash
# run_pheno.sh

for seed in 42 0 1 12 123; do
  for pair1 in "ihm-48-cxr-notes-ecg 2 f1" "los-48-cxr-notes-ecg 2 f1"; do
    read task num_labels primary_metric <<< "$pair1"
    for pair2 in "3 2"; do
      read experts top_k <<< "$pair2"
      for pair3 in "True True False" "True False False"; do
        read noisy normalized use_bias <<< "$pair3"

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
          --use_bias "$use_bias"
      done
    done
  done
done

