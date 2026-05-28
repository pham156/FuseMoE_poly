#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J proto_org26
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_26/%j_proto_org26.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_26/%j_proto_org26.err

set -eo pipefail

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

cd /home/pham156/MoE/FuseMoE_poly/src/scripts
mkdir -p /home/pham156/MoE/FuseMoE_poly/out/Week_26/prototype_router

SEED=30
RATIO=1.0
SAFE_RATIO="1p0"
OUTPUT_DIR="/home/pham156/MoE/FuseMoE_poly/run_folder/week26/prototype_dense_organ_supervised_ecg/seed${SEED}_ratio_${SAFE_RATIO}"
DIAG_PATH="/home/pham156/MoE/FuseMoE_poly/out/Week_26/prototype_router/prototype_dense_organ_supervised_ecg_seed${SEED}_ratio_${SAFE_RATIO}_router_diagnostics.csv"
rm -f "$DIAG_PATH"

echo "===== WEEK26 PROTOTYPE ORGAN-SUPERVISED ROUTER START ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "task: ihm-48-cxr-notes-ecg"
echo "modeltype: TS_CXR_Text_ECG"
echo "num_modalities: 4"
echo "seed: ${SEED}"
echo "ratio: ${RATIO}"
echo "epochs: 8"
echo "gating_function: poly"
echo "router: joint prototype dense"
echo "experts: e0 cardiovascular, e1 respiratory, e2 renal_metabolic, e3 infection"
echo "weak organ supervision: on"
echo "router_diagnostics_path: $DIAG_PATH"
echo "=========================================================="

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
  --top_k 4 \
  --disjoint_top_k 2 \
  --hidden_size 512 \
  --use_pt_text_embeddings \
  --router_type "joint" \
  --reg_ts \
  --use_balance_loss \
  --balance_loss_coef 1.0 \
  --noisy_gating True \
  --shared_experts 1 \
  --use_bias False \
  --use_temp False \
  --normalized True \
  --use_prototype_router \
  --prototype_router_dim 256 \
  --prototype_router_temperature 1.0 \
  --prototype_router_dense \
  --prototype_router_orth_coef 0.01 \
  --use_router_organ_supervision \
  --router_organ_supervision_coef 0.2 \
  --log_router_diagnostics \
  --router_diagnostics_layers all \
  --router_diagnostics_path "$DIAG_PATH" \
  --router_diagnostics_max_text_chars 1200

echo "Finished Week26 prototype organ-supervised router run"
