#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J fuse_bal8grid_%j
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_26/fuse_bal8grid_%j.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_26/fuse_bal8grid_%j.err

set -eo pipefail

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

cd /home/pham156/MoE/FuseMoE_poly/src/scripts
mkdir -p /home/pham156/MoE/FuseMoE_poly/out/Week_26

RATIOS=(0.2 0.4 0.6 0.8 1.0)

for RATIO in "${RATIOS[@]}"; do
  OUTPUT_DIR="/home/pham156/MoE/FuseMoE_poly/run_folder/week26/fusemoe_balance_8epoch_grid/ihm_joint4_k2_seed30_ratio_${RATIO}"

  echo "===== WEEK26 FUSEMOE BALANCE 8EPOCH GRID START ====="
  echo "job_id: ${SLURM_JOB_ID:-N/A}"
  echo "task: ihm-48-cxr-notes-ecg"
  echo "ratio: $RATIO"
  echo "seed: 30"
  echo "epochs: 8"
  echo "router_type: joint"
  echo "experts: 4"
  echo "top_k: 2"
  echo "output_dir: $OUTPUT_DIR"
  echo "===================================================="

  python -u -W ignore main_mimiciv.py \
    --num_train_epochs 8 \
    --modeltype 'TS_CXR_Text' \
    --kernel_size 1 \
    --train_batch_size 2 \
    --eval_batch_size 8 \
    --seed 30 \
    --gradient_accumulation_steps 16 \
    --num_update_bert_epochs 2 \
    --bertcount 0 \
    --ts_learning_rate 0.0004 \
    --txt_learning_rate 0.00002 \
    --notes_order 'Last' \
    --num_of_notes 5 \
    --max_length 1024 \
    --layers 3 \
    --output_dir "$OUTPUT_DIR" \
    --disable_run_folder_save \
    --embed_dim 128 \
    --num_modalities 3 \
    --model_name "bioLongformer" \
    --dataset "mimic" \
    --task "ihm-48-cxr-notes-ecg" \
    --primary_metric "f1" \
    --file_path '../../data/MIMIC-IV' \
    --num_labels 2 \
    --train_sample_ratio "$RATIO" \
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
    --gating_function "softmax" \
    --num_of_experts 4 \
    --top_k 2 \
    --disjoint_top_k 2 \
    --hidden_size 512 \
    --use_pt_text_embeddings \
    --router_type "joint" \
    --reg_ts \
    --use_balance_loss \
    --balance_loss_coef 0.01 \
    --noisy_gating True \
    --shared_experts 0 \
    --use_bias False \
    --use_temp False \
    --normalized True

  echo "Finished FuseMoE balance 8epoch ratio=$RATIO"
  echo "----------------------------------------------------"
done
