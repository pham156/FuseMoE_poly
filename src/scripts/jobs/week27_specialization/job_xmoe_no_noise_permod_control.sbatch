#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J xmoe_non
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_27/xmoe_noise_control/%j_%x.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_27/xmoe_noise_control/%j_%x.err

set -eo pipefail

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

cd /home/pham156/MoE/FuseMoE_poly/src/scripts
mkdir -p /home/pham156/MoE/FuseMoE_poly/out/Week_27/xmoe_noise_control

TASK="${TASK:-ihm-48-cxr-notes-ecg}"
NUM_LABELS="${NUM_LABELS:-2}"
PRIMARY_METRIC="${PRIMARY_METRIC:-f1}"
SEED="${SEED:-32}"
EPOCHS="${EPOCHS:-16}"
GATING_FUNCTION="${GATING_FUNCTION:-softmax}"
XMOE_NOISE_SCALE="${XMOE_NOISE_SCALE:-0.0}"
BALANCE_COEF="${BALANCE_COEF:-0.01}"
MODELTYPE="${MODELTYPE:-TS_CXR_Text}"
NUM_MODALITIES="${NUM_MODALITIES:-3}"

TAG="xmoe_no_noise_${TASK}_s${SEED}_${GATING_FUNCTION}_permod_ep${EPOCHS}_xn${XMOE_NOISE_SCALE}"
OUTPUT_DIR="/home/pham156/MoE/FuseMoE_poly/run_folder/week27/xmoe_noise_control/${TAG}"

echo "===== WEEK27 XMOE NO-NOISE PERMOD CONTROL ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "tag: ${TAG}"
echo "task: ${TASK}"
echo "seed: ${SEED}"
echo "epochs: ${EPOCHS}"
echo "gating: ${GATING_FUNCTION}"
echo "router: permod"
echo "use_xmoe_router: True"
echo "xmoe_noise_scale: ${XMOE_NOISE_SCALE}"
echo "balance_loss_coef: ${BALANCE_COEF}"
echo "modeltype: ${MODELTYPE}"
echo "num_modalities: ${NUM_MODALITIES}"
echo "output_dir: ${OUTPUT_DIR}"
echo "================================================"

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
  --router_type permod \
  --reg_ts \
  --use_balance_loss \
  --balance_loss_coef "$BALANCE_COEF" \
  --noisy_gating True \
  --use_xmoe_router \
  --xmoe_router_dim 128 \
  --xmoe_noise_scale "$XMOE_NOISE_SCALE" \
  --shared_experts 1 \
  --use_bias False \
  --use_temp False \
  --normalized True \
  --router_temperature 1.0 \
  --router_print_mode concise)

"${cmd[@]}"

echo "Finished ${TAG}"
