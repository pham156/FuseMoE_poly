#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J dsgrid27
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_27/deepseek_grid/%j_%x.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_27/deepseek_grid/%j_%x.err

set -eo pipefail

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

cd /home/pham156/MoE/FuseMoE_poly/src/scripts

CONFIG_ID="${CONFIG_ID:-manual}"
TASK="${TASK:-ihm-48-cxr-notes-ecg}"
NUM_LABELS="${NUM_LABELS:-2}"
PRIMARY_METRIC="${PRIMARY_METRIC:-f1}"
SEED="${SEED:-32}"
GATING_FUNCTION="${GATING_FUNCTION:-softmax}"
USE_XMOE_ROUTER="${USE_XMOE_ROUTER:-False}"
ROUTER="${ROUTER:-joint}"
SHARED_EXPERTS="${SHARED_EXPERTS:-1}"
NOISE_REGIME="${NOISE_REGIME:-none}"

MODELTYPE="${MODELTYPE:-TS_CXR_Text}"
NUM_MODALITIES="${NUM_MODALITIES:-3}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-16}"
NUM_EXPERTS="${NUM_EXPERTS:-4}"
TOP_K="${TOP_K:-2}"
DISJOINT_TOP_K="${DISJOINT_TOP_K:-2}"
BALANCE_COEF="${BALANCE_COEF:-0.01}"
XMOE_ROUTER_DIM="${XMOE_ROUTER_DIM:-128}"
ROUTER_PRINT_MODE="${ROUTER_PRINT_MODE:-concise}"

if [[ "$NOISE_REGIME" == "none" ]]; then
  NOISY_GATING="False"
  ROUTER_NOISE_SCALE="0.0"
  XMOE_NOISE_SCALE="0.0"
elif [[ "$NOISE_REGIME" == "original" ]]; then
  NOISY_GATING="True"
  ROUTER_NOISE_SCALE="1.0"
  XMOE_NOISE_SCALE="1.0"
else
  echo "Unknown NOISE_REGIME=${NOISE_REGIME}; expected none or original" >&2
  exit 2
fi

OUTPUT_DIR="/home/pham156/MoE/FuseMoE_poly/run_folder/week27/deepseek_grid/${CONFIG_ID}"

echo "===== WEEK27 DEEPSEEK-INSPIRED GRID CONFIG START ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "config_id: ${CONFIG_ID}"
echo "task: ${TASK}"
echo "seed: ${SEED}"
echo "gating: ${GATING_FUNCTION}"
echo "use_xmoe_router: ${USE_XMOE_ROUTER}"
echo "router: ${ROUTER}"
echo "shared_experts: ${SHARED_EXPERTS}"
echo "noise_regime: ${NOISE_REGIME}"
echo "noisy_gating: ${NOISY_GATING}"
echo "router_noise_scale: ${ROUTER_NOISE_SCALE}"
echo "xmoe_noise_scale: ${XMOE_NOISE_SCALE}"
echo "epochs: ${TOTAL_EPOCHS}"
echo "balance_loss_coef: ${BALANCE_COEF}"
echo "======================================================"

cmd=(python -u -W ignore main_mimiciv.py \
  --num_train_epochs "$TOTAL_EPOCHS" \
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
  --cross_method "moe" \
  --gating_function "$GATING_FUNCTION" \
  --num_of_experts "$NUM_EXPERTS" \
  --top_k "$TOP_K" \
  --disjoint_top_k "$DISJOINT_TOP_K" \
  --hidden_size 512 \
  --use_pt_text_embeddings \
  --router_type "$ROUTER" \
  --reg_ts \
  --use_balance_loss \
  --balance_loss_coef "$BALANCE_COEF" \
  --noisy_gating "$NOISY_GATING" \
  --router_noise_scale "$ROUTER_NOISE_SCALE" \
  --shared_experts "$SHARED_EXPERTS" \
  --use_bias False \
  --use_temp False \
  --normalized True \
  --router_temperature 1.0 \
  --router_print_mode "$ROUTER_PRINT_MODE")

if [[ "$USE_XMOE_ROUTER" == "True" || "$USE_XMOE_ROUTER" == "true" || "$USE_XMOE_ROUTER" == "1" ]]; then
  cmd+=(--use_xmoe_router --xmoe_router_dim "$XMOE_ROUTER_DIM" --xmoe_noise_scale "$XMOE_NOISE_SCALE")
fi

"${cmd[@]}"

echo "Finished Week27 DeepSeek-inspired config ${CONFIG_ID}"
