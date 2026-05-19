#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J sem_repl26
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_26/%j_sem_repl26.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_26/%j_sem_repl26.err

set -eo pipefail

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

cd /home/pham156/MoE/FuseMoE_poly/src/scripts
mkdir -p /home/pham156/MoE/FuseMoE_poly/out/Week_26

OUTPUT_DIR="/home/pham156/MoE/FuseMoE_poly/run_folder/week26/semantic_poly_note_replace/full_seed30"
DIAG_PATH="/home/pham156/MoE/FuseMoE_poly/out/Week_26/semantic_poly_note_replace_seed30_router_diagnostics.csv"

echo "===== WEEK26 SEMANTIC POLY NOTE REPLACE START ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "task: ihm-48-cxr-notes-ecg"
echo "ratio: 1.0"
echo "seed: 30"
echo "epochs: 8"
echo "gating_function: poly"
echo "normalized: True"
echo "semantic_profile_source: note"
echo "semantic_profile_layers: first"
echo "semantic_profile_fusion: replace"
echo "semantic_profile_kernel: 1/(1 + sqrt(2 - 2*cosine)^poly_power)"
echo "router_diagnostics_path: $DIAG_PATH"
echo "==================================================="

rm -f "$DIAG_PATH"

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
  --train_sample_ratio 1.0 \
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
  --poly_power 2 \
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
  --normalized True \
  --use_semantic_expert_profiles \
  --semantic_profile_set icu_organ_system \
  --semantic_profile_source note \
  --semantic_profile_note_pooling max \
  --semantic_profile_layers first \
  --semantic_profile_fusion replace \
  --semantic_profile_scale 1.0 \
  --log_router_diagnostics \
  --router_diagnostics_layers all \
  --router_diagnostics_path "$DIAG_PATH" \
  --router_diagnostics_max_text_chars 1200

echo "Finished semantic poly note replace run"
