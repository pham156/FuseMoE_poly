#!/bin/bash
set -euo pipefail

SCRIPT_ROOT="/home/pham156/MoE/FuseMoE_poly/src/scripts"
OLD_EXPERIMENT_DIR="$SCRIPT_ROOT/old_experiments"
cd "$SCRIPT_ROOT"

export CUDA_VISIBLE_DEVICES=0

# --- student_t gating noise--- 
for router in joint permod; do
  for degree in 0.3 0.5 1.5 2 4; do
    for pair2 in "4 2"; do
      read experts top_k <<< "$pair2"
      echo "Running gating=student_noise, student_degree=$degree, router_type=$router, num_experts=$experts, top_k=$top_k"

      python -W ignore main_mimiciv.py \
        --num_train_epochs 8 \
        --modeltype 'TS_CXR_Text' \
        --kernel_size 1 \
        --train_batch_size 2 \
        --eval_batch_size 8 \
        --seed 42 \
        --gradient_accumulation_steps 16 \
        --num_update_bert_epochs 2 \
        --bertcount 0 \
        --ts_learning_rate 0.0004 \
        --txt_learning_rate 0.00002 \
        --notes_order 'Last' \
        --num_of_notes 5 \
        --max_length 1024 \
        --layers 3 \
        --output_dir "../run_folder/week3/run_mimiciv_poly/TS_CXR_Text" \
        --embed_dim 128 \
        --num_modalities 3 \
        --model_name "bioLongformer" \
        --task 'ihm-48-cxr-notes-ecg' \
        --file_path '../../data/MIMIC-IV' \
        --num_labels 2 \
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
        --gating_function "student_t" \
        --student_degree "$degree" \
        --num_of_experts "$experts" \
        --top_k "$top_k" \
        --disjoint_top_k "$top_k" \
        --hidden_size 512 \
        --use_pt_text_embeddings \
        --router_type "$router" \
        --reg_ts \
        --use_balance_loss \
        --balance_loss_coef 0.01 \
        --noisy_gating True

      echo "Finished stduent_degree=$degree, router_type=$router, num_experts=$experts, top_k=$top_k"
      echo "---------------------------------------------"
    done
  done
done


# --- laplace gating noise --- #
# for router in joint permod; do
#   echo "Running gating=laplace_noise, router_type=$router"

#   python -W ignore main_mimiciv.py \
#     --num_train_epochs 8 \
#     --modeltype 'TS_CXR_Text' \
#     --kernel_size 1 \
#     --train_batch_size 2 \
#     --eval_batch_size 8 \
#     --seed 42 \
#     --gradient_accumulation_steps 16 \
#     --num_update_bert_epochs 2 \
#     --bertcount 0 \
#     --ts_learning_rate 0.0004 \
#     --txt_learning_rate 0.00002 \
#     --notes_order 'Last' \
#     --num_of_notes 5 \
#     --max_length 1024 \
#     --layers 3 \
#     --output_dir "../run_folder/week3/run_mimiciv_poly/TS_CXR_Text" \
#     --embed_dim 128 \
#     --num_modalities 3 \
#     --model_name "bioLongformer" \
#     --task 'ihm-48-cxr-notes-ecg' \
#     --file_path '/workspace/FuseMoE/data/MIMIC-IV' \
#     --num_labels 2 \
#     --num_heads 8 \
#     --embed_time 64 \
#     --tt_max 48 \
#     --TS_mixup \
#     --mixup_level 'batch' \
#     --fp16 \
#     --irregular_learn_emb_text \
#     --irregular_learn_emb_ts \
#     --irregular_learn_emb_cxr \
#     --irregular_learn_emb_ecg \
#     --cross_method "moe" \
#     --gating_function "laplace" \
#     --num_of_experts 16 \
#     --top_k 4 \
#     --disjoint_top_k 2 \
#     --hidden_size 512 \
#     --use_pt_text_embeddings \
#     --router_type "$router" \
#     --reg_ts \
#     --use_balance_loss \
#     --balance_loss_coef 0.01 \
#     --noisy_gating True

#   echo "Finished router_type=$router"
#   echo "---------------------------------------------"
# done

# --- softmax gating noise --- #
# for router in joint permod; do
#   echo "Running gating=softmax_noise, router_type=$router"

#   python -W ignore main_mimiciv.py \
#     --num_train_epochs 8 \
#     --modeltype 'TS_CXR_Text' \
#     --kernel_size 1 \
#     --train_batch_size 2 \
#     --eval_batch_size 8 \
#     --seed 42 \
#     --gradient_accumulation_steps 16 \
#     --num_update_bert_epochs 2 \
#     --bertcount 0 \
#     --ts_learning_rate 0.0004 \
#     --txt_learning_rate 0.00002 \
#     --notes_order 'Last' \
#     --num_of_notes 5 \
#     --max_length 1024 \
#     --layers 3 \
#     --output_dir "../run/TS_CXR_Text" \
#     --embed_dim 128 \
#     --num_modalities 3 \
#     --model_name "bioLongformer" \
#     --task 'ihm-48-cxr-notes-ecg' \
#     --file_path '/workspace/FuseMoE/data/MIMIC-IV' \
#     --num_labels 2 \
#     --num_heads 8 \
#     --embed_time 64 \
#     --tt_max 48 \
#     --TS_mixup \
#     --mixup_level 'batch' \
#     --fp16 \
#     --irregular_learn_emb_text \
#     --irregular_learn_emb_ts \
#     --irregular_learn_emb_cxr \
#     --irregular_learn_emb_ecg \
#     --cross_method "moe" \
#     --gating_function "softmax" \
#     --num_of_experts 16 \
#     --top_k 4 \
#     --disjoint_top_k 2 \
#     --hidden_size 512 \
#     --use_pt_text_embeddings \
#     --router_type "$router" \
#     --reg_ts \
#     --use_balance_loss \
#     --balance_loss_coef 0.01 \
#     --noisy_gating True

#   echo "Finished router_type=$router"
#   echo "---------------------------------------------"
# done
