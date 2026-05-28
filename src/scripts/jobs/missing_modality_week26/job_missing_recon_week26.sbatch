#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J miss_recon26
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_26/%j_miss_recon26.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_26/%j_miss_recon26.err

set -eo pipefail

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

cd /home/pham156/MoE/FuseMoE_poly/src/scripts
mkdir -p /home/pham156/MoE/FuseMoE_poly/out/Week_26/missing_modality

SEED=30
RATIO=1.0
SAFE_RATIO="1p0"
COEF=0.05
SAFE_COEF="0p05"

OUTPUT_DIR="/home/pham156/MoE/FuseMoE_poly/run_folder/week26/missing_recon_ecg/seed${SEED}_ratio_${SAFE_RATIO}_coef_${SAFE_COEF}"
DIAG_PATH="/home/pham156/MoE/FuseMoE_poly/out/Week_26/missing_modality/missing_recon_ecg_seed${SEED}_ratio_${SAFE_RATIO}_coef_${SAFE_COEF}_router_diagnostics.csv"
rm -f "$DIAG_PATH"

echo "===== WEEK26 MISSING-MODALITY RECONSTRUCTION START ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "task: ihm-48-cxr-notes-ecg"
echo "modeltype: TS_CXR_Text_ECG"
echo "num_modalities: 4"
echo "seed: ${SEED}"
echo "ratio: ${RATIO}"
echo "missing_recon_coef: ${COEF}"
echo "missing_recon_targets: cxr,ecg,text"
echo "output_dir: ${OUTPUT_DIR}"
echo "router_diagnostics_path: ${DIAG_PATH}"
echo "======================================================="

python -u -W ignore main_mimiciv.py \
  --num_train_epochs 8 \
  --modeltype 'TS_CXR_Text_ECG' \
  --kernel_size 1 \
  --train_batch_size 2 \
  --eval_batch_size 8 \
  --seed "$SEED" \
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
  --num_modalities 4 \
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
  --balance_loss_coef 1.0 \
  --noisy_gating True \
  --shared_experts 0 \
  --use_bias False \
  --use_temp False \
  --normalized True \
  --use_missing_modality_recon \
  --missing_modality_recon_coef "$COEF" \
  --missing_modality_recon_targets "cxr,ecg,text" \
  --missing_modality_recon_hidden 256 \
  --log_router_diagnostics \
  --router_diagnostics_layers all \
  --router_diagnostics_path "$DIAG_PATH" \
  --router_diagnostics_max_text_chars 1200

echo "Finished Week26 missing-modality reconstruction run"
