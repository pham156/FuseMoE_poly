#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

# ---- Laplace-Laplace gating (baseline) ----
for router in joint; do
  echo "Running gating=laplace-laplace, router_type=$router"

  python -W ignore main_mimiciv.py \
    --num_train_epochs 8 \
    --modeltype 'TS_CXR_Text' \
    --kernel_size 1 \
    --train_batch_size 1 \
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
    --output_dir "../run_folder/run_mimiciv_hme/TS_CXR_Text" \
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
    --cross_method "hme" \
    --gating_function "laplace" "laplace"\
    --num_of_experts 3 5 \
    --top_k 2 4 \
    --disjoint_top_k 2 \
    --hidden_size 512 \
    --use_pt_text_embeddings \
    --router_type "$router" \
    --reg_ts \
    --use_balance_loss \
    --balance_loss_coef 0.01

  echo "Finished gating=laplace-laplace, router_type=$router"
  echo "---------------------------------------------"
done

# ---- Softmax-Softmax gating (baseline) ----
for router in joint; do
  echo "Running gating=laplace-laplace, router_type=$router"

  python -W ignore main_mimiciv.py \
    --num_train_epochs 8 \
    --modeltype 'TS_CXR_Text' \
    --kernel_size 1 \
    --train_batch_size 1 \
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
    --output_dir "../run_folder/run_mimiciv_hme/TS_CXR_Text" \
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
    --cross_method "hme" \
    --gating_function "softmax" "softmax"\
    --num_of_experts 3 5 \
    --top_k 2 4 \
    --disjoint_top_k 2 \
    --hidden_size 512 \
    --use_pt_text_embeddings \
    --router_type "$router" \
    --reg_ts \
    --use_balance_loss \
    --balance_loss_coef 0.01

  echo "Finished gating=softmax-softmax, router_type=$router"
  echo "---------------------------------------------"
done


# ---- Softmax-Laplace gating (baseline) ----
for router in joint; do
  echo "Running gating=softmax-laplace, router_type=$router"

  python -W ignore main_mimiciv.py \
    --num_train_epochs 8 \
    --modeltype 'TS_CXR_Text' \
    --kernel_size 1 \
    --train_batch_size 1 \
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
    --output_dir "../run_folder/run_mimiciv_hme/TS_CXR_Text" \
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
    --cross_method "hme" \
    --gating_function "softmax" "laplace"\
    --num_of_experts 3 5 \
    --top_k 2 4 \
    --disjoint_top_k 2 \
    --hidden_size 512 \
    --use_pt_text_embeddings \
    --router_type "$router" \
    --reg_ts \
    --use_balance_loss \
    --balance_loss_coef 0.01

  echo "Finished gating=softmax-laplace, router_type=$router"
  echo "---------------------------------------------"
done

# ---- laplace-softmax gating (baseline) ----

for router in joint; do
  echo "Running gating=softmax-laplace, router_type=$router"

  python -W ignore main_mimiciv.py \
    --num_train_epochs 8 \
    --modeltype 'TS_CXR_Text' \
    --kernel_size 1 \
    --train_batch_size 1 \
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
    --output_dir "../run_folder/run_mimiciv_hme/TS_CXR_Text" \
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
    --cross_method "hme" \
    --gating_function "laplace" "softmax"\
    --num_of_experts 3 5 \
    --top_k 2 4 \
    --disjoint_top_k 2 \
    --hidden_size 512 \
    --use_pt_text_embeddings \
    --router_type "$router" \
    --reg_ts \
    --use_balance_loss \
    --balance_loss_coef 0.01

  echo "Finished gating=laplace-softmax, router_type=$router"
  echo "---------------------------------------------"
done

# --- laplace-poly gating --- #
for router in joint permod; do
  for power in 0.3 0.5 1.5 2 2.5 3 4 5 6; do
    echo "Running gating=laplace-poly, poly_power=$power, router_type=$router"

    python -W ignore main_mimiciv.py \
      --num_train_epochs 8 \
      --modeltype 'TS_CXR_Text' \
      --kernel_size 1 \
      --train_batch_size 1 \
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
      --output_dir "../run_folder/run_mimiciv_hme/TS_CXR_Text" \
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
      --cross_method "hme" \
      --gating_function "laplace" "poly" \
      --poly_power "$power" \
      --num_of_experts 3 5 \
      --top_k 2 4 \
      --disjoint_top_k 2 \
      --hidden_size 512 \
      --use_pt_text_embeddings \
      --router_type "$router" \
      --reg_ts \
      --use_balance_loss \
      --balance_loss_coef 0.01

    echo "Finished poly_power=$power, router_type=$router"
    echo "---------------------------------------------"
  done
done


# --- softmax-poly gating --- #
for router in joint; do
  for power in 0.3 0.5 1.5 2 2.5 3 4 5 6; do
    echo "Running gating=softmax-poly, poly_power=$power, router_type=$router"

    python -W ignore main_mimiciv.py \
      --num_train_epochs 8 \
      --modeltype 'TS_CXR_Text' \
      --kernel_size 1 \
      --train_batch_size 1 \
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
      --output_dir "../run_folder/run_mimiciv_hme/TS_CXR_Text" \
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
      --cross_method "hme" \
      --gating_function "softmax" "poly" \
      --poly_power "$power" \
      --num_of_experts 3 5 \
      --top_k 2 4 \
      --disjoint_top_k 2 \
      --hidden_size 512 \
      --use_pt_text_embeddings \
      --router_type "$router" \
      --reg_ts \
      --use_balance_loss \
      --balance_loss_coef 0.01

    echo "Finished poly_power=$power, router_type=$router"
    echo "---------------------------------------------"
  done
done
