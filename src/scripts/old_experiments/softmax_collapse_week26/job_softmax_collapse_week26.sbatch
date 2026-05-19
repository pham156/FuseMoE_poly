#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J soft_col26
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_26/%j_soft_col26.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_26/%j_soft_col26.err

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
ROUTER="${ROUTER:-joint}"
EXPERTS="${EXPERTS:-4}"
TOP_K="${TOP_K:-2}"
DISJOINT_TOP_K="${DISJOINT_TOP_K:-2}"

# Keep this as an old-config collapse diagnostic:
# - no semantic profiles
# - no prototype router
# - no weak organ supervision
# - no missing-modality reconstruction
# - no Lingshu path
BALANCE_COEFS=(${BALANCE_COEFS:-0.01 0.1 1.0})
NOISY_OPTIONS=(${NOISY_OPTIONS:-True})

echo "===== WEEK26 OLD-CONFIG SOFTMAX COLLAPSE START ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "task: ${TASK}"
echo "modeltype: TS_CXR_Text"
echo "num_modalities: 3"
echo "seed: ${SEED}"
echo "ratio: ${RATIO}"
echo "router: ${ROUTER}"
echo "experts: ${EXPERTS}"
echo "top_k: ${TOP_K}"
echo "gating: softmax"
echo "balance_coefs: ${BALANCE_COEFS[*]}"
echo "noisy_options: ${NOISY_OPTIONS[*]}"
echo "===================================================="

for BALANCE_COEF in "${BALANCE_COEFS[@]}"; do
  for NOISY in "${NOISY_OPTIONS[@]}"; do
    SAFE_BAL="${BALANCE_COEF/./p}"
    SAFE_NOISY="$(echo "$NOISY" | tr '[:upper:]' '[:lower:]')"
    OUTPUT_DIR="/home/pham156/MoE/FuseMoE_poly/run_folder/week26/softmax_collapse_oldcfg/${TASK}_${ROUTER}${EXPERTS}_k${TOP_K}_seed${SEED}_ratio${SAFE_RATIO}_bal${SAFE_BAL}_noisy${SAFE_NOISY}"
    DIAG_PATH="/home/pham156/MoE/FuseMoE_poly/out/Week_26/softmax_collapse/softmax_oldcfg_${TASK}_${ROUTER}${EXPERTS}_k${TOP_K}_seed${SEED}_ratio${SAFE_RATIO}_bal${SAFE_BAL}_noisy${SAFE_NOISY}_router_diagnostics.csv"
    rm -f "$DIAG_PATH"

    echo ""
    echo "===== RUN softmax oldcfg balance=${BALANCE_COEF} noisy=${NOISY} ====="
    echo "output_dir: $OUTPUT_DIR"
    echo "router_diagnostics_path: $DIAG_PATH"

    python -u -W ignore main_mimiciv.py \
      --num_train_epochs 8 \
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
      --num_of_experts "$EXPERTS" \
      --top_k "$TOP_K" \
      --disjoint_top_k "$DISJOINT_TOP_K" \
      --hidden_size 512 \
      --use_pt_text_embeddings \
      --router_type "$ROUTER" \
      --reg_ts \
      --use_balance_loss \
      --balance_loss_coef "$BALANCE_COEF" \
      --noisy_gating "$NOISY" \
      --shared_experts 0 \
      --use_bias False \
      --use_temp False \
      --normalized True \
      --log_router_diagnostics \
      --router_diagnostics_layers all \
      --router_diagnostics_path "$DIAG_PATH" \
      --router_diagnostics_max_text_chars 600

    echo "===== FINISHED softmax oldcfg balance=${BALANCE_COEF} noisy=${NOISY} ====="
  done
done

echo "Finished Week26 old-config softmax collapse diagnostic"
