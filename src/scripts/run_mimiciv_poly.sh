#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

# --- poly gating noise--- #
# for task in "ihm-48-cxr-notes-ecg 2 f1" "los-48-cxr-notes-ecg 2 f1" "pheno-all-cxr-notes-ecg 25 macro_f1"; do

for seed in 42; do
  for pair1 in "pheno-all-cxr-notes-ecg 25 macro_f1"; do
    read task num_labels primary_metric <<< "$pair1"
    for router in "permod"; do
      for power in 6 8; do
        for pair2 in "8 4"; do
          read experts top_k <<< "$pair2"
          for pair3 in "True False False"; do
            read noisy normalized use_bias <<< "$pair3"
            echo "Running gating=poly, normalized=$normalized, use_bias=$use_bias, seed=$seed, poly_power=$power, router_type=$router, num_experts=$experts, top_k=$top_k, noisy_gating=$noisy, task=$task"
            python -W ignore main_mimiciv.py \
              --num_train_epochs 8 \
              --modeltype 'TS_CXR_Text' \
              --kernel_size 1 \
              --train_batch_size 2 \
              --eval_batch_size 8 \
              --seed $seed \
              --gradient_accumulation_steps 16 \
              --num_update_bert_epochs 2 \
              --bertcount 0 \
              --ts_learning_rate 0.0004 \
              --txt_learning_rate 0.00002 \
              --notes_order 'Last' \
              --num_of_notes 5 \
              --max_length 1024 \
              --layers 3 \
              --output_dir "../../run_folder/week3/run_mimiciv_poly/TS_CXR_Text" \
              --embed_dim 128 \
              --num_modalities 3 \
              --model_name "bioLongformer" \
              --task "$task" \
              --primary_metric "$primary_metric" \
              --file_path '../../data/MIMIC-IV' \
              --num_labels "$num_labels" \
              --num_heads 8 \
              --embed_time 64 \
              --tt_max 48 \
              --TS_mixup \
              --mixup_level 'batch' \
              --fp16 \
              --irregular_learn_emb_text \
              --irregular_learn_emb_ts \
              --irregular_learn_emb_cxr \
              --irregular_learn_emb_ecg \
              --cross_method "moe" \
              --gating_function "poly" \
              --poly_power "$power" \
              --poly_powers 2 \
              --num_of_experts "$experts" \
              --top_k "$top_k" \
              --disjoint_top_k "$top_k" \
              --hidden_size 512 \
              --use_pt_text_embeddings \
              --router_type "$router" \
              --reg_ts \
              --use_balance_loss \
              --balance_loss_coef 0.01 \
              --noisy_gating "$noisy" \
              --normalized "$normalized" \
              --use_bias "$use_bias"

            echo "Finished poly_power=$power, normalized=$normalized, router_type=$router, num_experts=$experts, top_k=$top_k, noisy_gating=$noisy, task=$task"
            echo "---------------------------------------------"
          done
        done
      done
    done
  done
done



# # reproduce

# for seed in 42; do
#   for gating in "gaussian" "softmax" "laplace"; do
#     for router in joint permod; do
#       for pair2 in "4 2"; do
#         read experts top_k <<< "$pair2"
#         for noisy in True; do
#           echo "Running seed=$seed, gating=$gating, router_type=$router, num_experts=$experts, top_k=$top_k, noisy_gating=$noisy"
#           python -W ignore main_mimiciv.py \
#             --num_train_epochs 8 \
#             --modeltype 'TS_CXR_Text' \
#             --kernel_size 1 \
#             --train_batch_size 2 \
#             --eval_batch_size 8 \
#             --seed $seed \
#             --gradient_accumulation_steps 16 \
#             --num_update_bert_epochs 2 \
#             --bertcount 0 \
#             --ts_learning_rate 0.0004 \
#             --txt_learning_rate 0.00002 \
#             --notes_order 'Last' \
#             --num_of_notes 5 \
#             --max_length 1024 \
#             --layers 3 \
#             --output_dir "../run/TS_CXR_Text" \
#             --embed_dim 128 \
#             --num_modalities 3 \
#             --model_name "bioLongformer" \
#             --task 'pheno-all-cxr-notes-ecg' \
#             --file_path '../../data/MIMIC-IV' \
#             --num_labels 25 \
#             --num_heads 8 \
#             --embed_time 64 \
#             --tt_max 48 \
#             --TS_mixup \
#             --mixup_level 'batch' \
#             --fp16 \
#             --irregular_learn_emb_text \
#             --irregular_learn_emb_ts \
#             --irregular_learn_emb_cxr \
#             --irregular_learn_emb_ecg \
#             --cross_method "moe" \
#             --gating_function "$gating" \
#             --num_of_experts "$experts" \
#             --top_k "$top_k" \
#             --disjoint_top_k "$top_k" \
#             --hidden_size 512 \
#             --use_pt_text_embeddings \
#             --router_type "$router" \
#             --reg_ts \
#             --use_balance_loss \
#             --balance_loss_coef 0.01 \
#             --noisy_gating "$noisy" \
#             --primary_metric "macro_f1" \
#             --normalized True
#         done
#       done
#     done
#   done
# done