#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J spec27
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_27/specialization/%j_%x.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_27/specialization/%j_%x.err

set -eo pipefail

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

cd /home/pham156/MoE/FuseMoE_poly/src/scripts
mkdir -p /home/pham156/MoE/FuseMoE_poly/out/Week_27/specialization

TASK="${TASK:-ihm-48-cxr-notes-ecg}"
NUM_LABELS="${NUM_LABELS:-2}"
PRIMARY_METRIC="${PRIMARY_METRIC:-f1}"
SEED="${SEED:-32}"
ROUTER="${ROUTER:-permod}"
EPOCHS="${EPOCHS:-8}"
GATING_FUNCTION="${GATING_FUNCTION:-softmax}"
MODELTYPE="${MODELTYPE:-TS_CXR_Text}"
NUM_MODALITIES="${NUM_MODALITIES:-3}"
BALANCE_COEF="${BALANCE_COEF:-1.0}"
ROUTER_NOISE_SCALE="${ROUTER_NOISE_SCALE:-0.2}"
ROUTER_NOISE_FINAL_SCALE="${ROUTER_NOISE_FINAL_SCALE:-0.02}"
ROUTER_NOISE_DECAY_EPOCHS="${ROUTER_NOISE_DECAY_EPOCHS:-8}"
ROUTER_VARIANCE_COEF="${ROUTER_VARIANCE_COEF:-0.001}"
OUTPUT_ORTH_COEF="${OUTPUT_ORTH_COEF:-0.001}"
SPECIALIZATION_LOSS_MODE="${SPECIALIZATION_LOSS_MODE:-paper}"
SPECIALIZATION_AUX_COEF="${SPECIALIZATION_AUX_COEF:-0.001}"
DENSE_ROUTING="${DENSE_ROUTING:-False}"

TAG="spec_${TASK}_s${SEED}_${ROUTER}_${GATING_FUNCTION}_ep${EPOCHS}_bal${BALANCE_COEF}_var${ROUTER_VARIANCE_COEF}_orth${OUTPUT_ORTH_COEF}_n${ROUTER_NOISE_SCALE}to${ROUTER_NOISE_FINAL_SCALE}"
OUTPUT_DIR="/home/pham156/MoE/FuseMoE_poly/run_folder/week27/specialization/${TAG}"

echo "===== WEEK27 SPECIALIZATION GRID START ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "tag: ${TAG}"
echo "task: ${TASK}"
echo "seed: ${SEED}"
echo "router: ${ROUTER}"
echo "epochs: ${EPOCHS}"
echo "gating: ${GATING_FUNCTION}"
echo "modeltype: ${MODELTYPE}"
echo "num_modalities: ${NUM_MODALITIES}"
echo "balance_coef_outer: ${BALANCE_COEF}"
echo "router_noise: ${ROUTER_NOISE_SCALE} -> ${ROUTER_NOISE_FINAL_SCALE} over ${ROUTER_NOISE_DECAY_EPOCHS} epochs"
echo "router_variance_coef: ${ROUTER_VARIANCE_COEF}"
echo "output_orth_coef: ${OUTPUT_ORTH_COEF}"
echo "specialization_loss_mode: ${SPECIALIZATION_LOSS_MODE}"
echo "specialization_aux_coef: ${SPECIALIZATION_AUX_COEF}"
echo "dense_routing: ${DENSE_ROUTING}"
echo "output_dir: ${OUTPUT_DIR}"
echo "==========================================="

cmd=(python -u -W ignore main_mimiciv.py \
  --num_train_epochs "$EPOCHS" \
  --modeltype "$MODELTYPE" \
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
  --num_modalities "$NUM_MODALITIES" \
  --model_name "bioLongformer" \
  --dataset "mimic" \
  --task "$TASK" \
  --primary_metric "$PRIMARY_METRIC" \
  --file_path '../../data/MIMIC-IV' \
  --num_labels "$NUM_LABELS" \
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
  --gating_function "$GATING_FUNCTION" \
  --num_of_experts 4 \
  --top_k 2 \
  --disjoint_top_k 2 \
  --hidden_size 512 \
  --use_pt_text_embeddings \
  --router_type "$ROUTER" \
  --reg_ts \
  --use_balance_loss \
  --balance_loss_coef "$BALANCE_COEF" \
  --noisy_gating True \
  --router_noise_scale "$ROUTER_NOISE_SCALE" \
  --router_noise_final_scale "$ROUTER_NOISE_FINAL_SCALE" \
  --router_noise_decay_epochs "$ROUTER_NOISE_DECAY_EPOCHS" \
  --router_variance_coef "$ROUTER_VARIANCE_COEF" \
  --output_orth_coef "$OUTPUT_ORTH_COEF" \
  --specialization_loss_mode "$SPECIALIZATION_LOSS_MODE" \
  --specialization_aux_coef "$SPECIALIZATION_AUX_COEF" \
  --shared_experts 1 \
  --use_bias False \
  --use_temp False \
  --normalized True \
  --router_print_mode concise)

if [[ "$DENSE_ROUTING" == "True" || "$DENSE_ROUTING" == "true" || "$DENSE_ROUTING" == "1" ]]; then
  cmd+=(--prototype_router_dense)
fi

"${cmd[@]}"

echo "Finished specialization grid ${TAG}"
