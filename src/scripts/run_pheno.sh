#!/bin/bash
# run_pheno.sh

for seed in 42; do
  for pair1 in "pheno-all-cxr-notes-ecg 25 macro_f1"; do
    read task num_labels primary_metric <<< "$pair1"
    for pair2 in "3 1 1"; do
      read experts top_k shared_experts <<< "$pair2"
      for pair3 in "True True" "True False"; do
        read noisy normalized <<< "$pair3"
        for router in "joint" "permod"; do
          
          # Submit one job for this combination
          sbatch job_one_exp.sbatch \
            --seed "$seed" \
            --task "$task" \
            --num_labels "$num_labels" \
            --primary_metric "$primary_metric" \
            --experts "$experts" \
            --top_k "$top_k" \
            --router "$router" \
            --noisy "$noisy" \
            --normalized "$normalized" \
            --shared_experts "$shared_experts"
        done
      done
    done
  done
done

