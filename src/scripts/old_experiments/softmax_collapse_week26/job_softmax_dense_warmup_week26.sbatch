#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J soft_warm26
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_26/%j_soft_warm26.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_26/%j_soft_warm26.err

set -eo pipefail

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

cd /home/pham156/MoE/FuseMoE_poly/src/scripts
mkdir -p /home/pham156/MoE/FuseMoE_poly/out/Week_26/softmax_collapse

TASK="${TASK:-ihm-48-cxr-notes-ecg}"
NUM_LABELS="${NUM_LABELS:-2}"
PRIMARY_METRIC="${PRIMARY_METRIC:-f1}"
SEED="${SEED:-30}"
RATIO="${RATIO:-1.0}"
SAFE_RATIO="${RATIO/./p}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-16}"
DENSE_WARMUP_EPOCHS="${DENSE_WARMUP_EPOCHS:-8}"
BALANCE_COEF="${BALANCE_COEF:-1.0}"
SHARED_EXPERTS="${SHARED_EXPERTS:-1}"
ROUTER_Z_LOSS_COEF="${ROUTER_Z_LOSS_COEF:-0.01}"
ROUTER_ENTROPY_COEF="${ROUTER_ENTROPY_COEF:-0.1}"

SAFE_BAL="${BALANCE_COEF/./p}"
SAFE_Z="${ROUTER_Z_LOSS_COEF/./p}"
SAFE_ENT="${ROUTER_ENTROPY_COEF/./p}"

OUTPUT_DIR="/home/pham156/MoE/FuseMoE_poly/run_folder/week26/softmax_dense_warmup/${TASK}_joint4_k2_seed${SEED}_ratio${SAFE_RATIO}_ep${TOTAL_EPOCHS}_warm${DENSE_WARMUP_EPOCHS}_bal${SAFE_BAL}_z${SAFE_Z}_ent${SAFE_ENT}"
DIAG_PATH="/home/pham156/MoE/FuseMoE_poly/out/Week_26/softmax_collapse/softmax_dense_warmup_${TASK}_joint4_k2_seed${SEED}_ratio${SAFE_RATIO}_ep${TOTAL_EPOCHS}_warm${DENSE_WARMUP_EPOCHS}_bal${SAFE_BAL}_z${SAFE_Z}_ent${SAFE_ENT}_router_diagnostics.csv"
rm -f "$DIAG_PATH"

echo "===== WEEK26 SOFTMAX DENSE WARMUP START ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "task: ${TASK}"
echo "modeltype: TS_CXR_Text"
echo "num_modalities: 3"
echo "seed: ${SEED}"
echo "ratio: ${RATIO}"
echo "router: joint"
echo "experts: 4"
echo "top_k after warmup: 2"
echo "gating: softmax"
echo "total_epochs: ${TOTAL_EPOCHS}"
echo "dense_warmup_epochs: ${DENSE_WARMUP_EPOCHS}"
echo "shared_experts: ${SHARED_EXPERTS}"
echo "balance_loss_coef: ${BALANCE_COEF}"
echo "router_z_loss_coef: ${ROUTER_Z_LOSS_COEF}"
echo "router_entropy_coef: ${ROUTER_ENTROPY_COEF}"
echo "output_dir: ${OUTPUT_DIR}"
echo "router_diagnostics_path: ${DIAG_PATH}"
echo "================================================"

python -u -W ignore main_mimiciv.py \
  --num_train_epochs "$TOTAL_EPOCHS" \
  --modeltype 'TS_CXR_Text' \
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
  --num_modalities 3 \
  --model_name "bioLongformer" \
  --dataset "mimic" \
  --task "$TASK" \
  --primary_metric "$PRIMARY_METRIC" \
  --file_path '../../data/MIMIC-IV' \
  --num_labels "$NUM_LABELS" \
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
  --balance_loss_coef "$BALANCE_COEF" \
  --noisy_gating True \
  --shared_experts "$SHARED_EXPERTS" \
  --use_bias False \
  --use_temp False \
  --normalized True \
  --router_z_loss_coef "$ROUTER_Z_LOSS_COEF" \
  --router_z_loss_type logsumexp \
  --router_entropy_coef "$ROUTER_ENTROPY_COEF" \
  --dense_warmup_epochs "$DENSE_WARMUP_EPOCHS" \
  --log_router_diagnostics \
  --router_diagnostics_layers all \
  --router_diagnostics_path "$DIAG_PATH" \
  --router_diagnostics_max_text_chars 600

echo "Finished Week26 softmax dense warmup"
