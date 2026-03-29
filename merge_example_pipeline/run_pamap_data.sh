#!/bin/bash

# python pamap_rus_multimodal.py \
#     --method batch \
#     --subject_id 1 \
#     --max_lag 10 \
#     --dominance_threshold 0.4 \
#     --dominance_percentage 0.9 \
#     --gpu 0 \
#     --hidden_dim 64 \
#     --layers 3 \
#     --lr 0.001 \
#     --discrim_epochs 30 \
#     --ce_epochs 15 \
#     --activation relu \
#     --embed_dim 20 \
#     --batch_size 512 \
#     --n_batches 3 \
#     --seed 42

python train_pamap_multimodal.py \
    --subject_id 1 \
    --seq_len 100 \
    --window_step 50 \
    --val_split 0.2 \
    --rus_max_lag 10 \
    --rus_bins 4 \
    --d_model 128 \
    --nhead 4 \
    --d_ff 256 \
    --num_encoder_layers 6 \
    --num_moe_layers 3 \
    --dropout 0.1 \
    --modality_encoder_layers 2 \
    --moe_num_experts 8 \
    --moe_num_synergy_experts 2 \
    --moe_k 2 \
    --moe_expert_hidden_dim 128 \
    --moe_capacity_factor 1.25 \
    --moe_router_gru_hidden_dim 64 \
    --moe_router_token_processed_dim 64 \
    --moe_router_attn_key_dim 32 \
    --moe_router_attn_value_dim 32 \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --clip_grad_norm 1.0 \
    --use_lr_scheduler \
    --threshold_u 0.5 \
    --threshold_r 0.1 \
    --threshold_s 0.1 \
    --lambda_u 10 \
    --lambda_r 10 \
    --lambda_s 10 \
    --epsilon_loss 1e-8 \
    --seed 42 \
    --cuda_device 1 \
    --wandb_project pamap-multimodal-trus-moe
