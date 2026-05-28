#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J dsarch27
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_27/deepseek_arch_grid/%j_%x.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_27/deepseek_arch_grid/%j_%x.err

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
ROUTER="${ROUTER:-joint}"
SHARED_EXPERTS="${SHARED_EXPERTS:-1}"
ARCH_VARIANT="${ARCH_VARIANT:-zero_original_noise}"

MODELTYPE="${MODELTYPE:-TS_CXR_Text}"
NUM_MODALITIES="${NUM_MODALITIES:-3}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-16}"
NUM_EXPERTS="${NUM_EXPERTS:-4}"
TOP_K="${TOP_K:-2}"
DISJOINT_TOP_K="${DISJOINT_TOP_K:-2}"
BALANCE_COEF="${BALANCE_COEF:-0.01}"
XMOE_ROUTER_DIM="${XMOE_ROUTER_DIM:-128}"
ROUTER_PRINT_MODE="${ROUTER_PRINT_MODE:-concise}"
LOG_EXPERT_OUTPUT_DIAGNOSTICS="${LOG_EXPERT_OUTPUT_DIAGNOSTICS:-False}"

USE_XMOE_ROUTER="False"
ROUTER_INIT="zero"
ROUTER_INIT_STD="0.02"
NOISY_GATING="True"
ROUTER_NOISE_SCALE="1.0"
ROUTER_NOISE_FINAL_SCALE=""
ROUTER_NOISE_DECAY_EPOCHS="0"
XMOE_NOISE_SCALE="1.0"
USE_BIAS="False"
ROUTER_BIAS_MODE="mul"
GATE_NORMALIZATION="selected"
ROUTER_TOPK_MODE="k_plus_1"
MOE_MIXING_SPACE="logprob"
LOAD_BALANCE_MODE="cv"

case "$ARCH_VARIANT" in
  fusemoe_original)
    ;;
  zero_no_noise)
    NOISY_GATING="False"
    ROUTER_NOISE_SCALE="0.0"
    ROUTER_TOPK_MODE="k"
    ;;
  random_router)
    ROUTER_INIT="xavier"
    NOISY_GATING="False"
    ROUTER_NOISE_SCALE="0.0"
    ROUTER_TOPK_MODE="k"
    ;;
  random_low_noise_decay)
    ROUTER_INIT="xavier"
    NOISY_GATING="True"
    ROUTER_NOISE_SCALE="0.1"
    ROUTER_NOISE_FINAL_SCALE="0.0"
    ROUTER_NOISE_DECAY_EPOCHS="8"
    ;;
  deepseek_gate)
    ROUTER_INIT="xavier"
    NOISY_GATING="False"
    ROUTER_NOISE_SCALE="0.0"
    ROUTER_TOPK_MODE="k"
    USE_BIAS="True"
    ROUTER_BIAS_MODE="add"
    GATE_NORMALIZATION="full_renorm"
    ;;
  deepseek_aux)
    ROUTER_INIT="xavier"
    NOISY_GATING="False"
    ROUTER_NOISE_SCALE="0.0"
    ROUTER_TOPK_MODE="k"
    USE_BIAS="True"
    ROUTER_BIAS_MODE="add"
    GATE_NORMALIZATION="full_renorm"
    LOAD_BALANCE_MODE="deepseek_aux"
    ;;
  deepseek_hidden_aux)
    ROUTER_INIT="xavier"
    NOISY_GATING="False"
    ROUTER_NOISE_SCALE="0.0"
    ROUTER_TOPK_MODE="k"
    USE_BIAS="True"
    ROUTER_BIAS_MODE="add"
    GATE_NORMALIZATION="full_renorm"
    MOE_MIXING_SPACE="linear"
    LOAD_BALANCE_MODE="deepseek_aux"
    ;;
  xmoe_hidden_aux)
    USE_XMOE_ROUTER="True"
    NOISY_GATING="False"
    ROUTER_TOPK_MODE="k"
    GATE_NORMALIZATION="full_renorm"
    MOE_MIXING_SPACE="linear"
    LOAD_BALANCE_MODE="deepseek_aux"
    XMOE_NOISE_SCALE="0.0"
    ;;
  *)
    echo "Unknown ARCH_VARIANT=${ARCH_VARIANT}" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="/home/pham156/MoE/FuseMoE_poly/run_folder/week27/deepseek_arch_grid/${CONFIG_ID}"

echo "===== WEEK27 DEEPSEEK ARCH GRID CONFIG START ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "config_id: ${CONFIG_ID}"
echo "task: ${TASK}"
echo "seed: ${SEED}"
echo "router: ${ROUTER}"
echo "shared_experts: ${SHARED_EXPERTS}"
echo "arch_variant: ${ARCH_VARIANT}"
echo "use_xmoe_router: ${USE_XMOE_ROUTER}"
echo "router_init: ${ROUTER_INIT}"
echo "noisy_gating: ${NOISY_GATING}"
echo "router_noise_scale: ${ROUTER_NOISE_SCALE}"
echo "router_noise_final_scale: ${ROUTER_NOISE_FINAL_SCALE:-none}"
echo "router_noise_decay_epochs: ${ROUTER_NOISE_DECAY_EPOCHS}"
echo "xmoe_noise_scale: ${XMOE_NOISE_SCALE}"
echo "use_bias: ${USE_BIAS}"
echo "router_bias_mode: ${ROUTER_BIAS_MODE}"
echo "gate_normalization: ${GATE_NORMALIZATION}"
echo "router_topk_mode: ${ROUTER_TOPK_MODE}"
echo "moe_mixing_space: ${MOE_MIXING_SPACE}"
echo "load_balance_mode: ${LOAD_BALANCE_MODE}"
echo "log_expert_output_diagnostics: ${LOG_EXPERT_OUTPUT_DIAGNOSTICS}"
echo "=================================================="

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
  --gating_function "softmax" \
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
  --router_init "$ROUTER_INIT" \
  --router_init_std "$ROUTER_INIT_STD" \
  --router_noise_scale "$ROUTER_NOISE_SCALE" \
  --shared_experts "$SHARED_EXPERTS" \
  --use_bias "$USE_BIAS" \
  --router_bias_mode "$ROUTER_BIAS_MODE" \
  --gate_normalization "$GATE_NORMALIZATION" \
  --router_topk_mode "$ROUTER_TOPK_MODE" \
  --moe_mixing_space "$MOE_MIXING_SPACE" \
  --load_balance_mode "$LOAD_BALANCE_MODE" \
  --use_temp False \
  --normalized True \
  --router_temperature 1.0 \
  --router_print_mode "$ROUTER_PRINT_MODE")

if [[ "$LOG_EXPERT_OUTPUT_DIAGNOSTICS" == "True" ]]; then
  cmd+=(--log_expert_output_diagnostics)
fi

if [[ -n "$ROUTER_NOISE_FINAL_SCALE" ]]; then
  cmd+=(--router_noise_final_scale "$ROUTER_NOISE_FINAL_SCALE" --router_noise_decay_epochs "$ROUTER_NOISE_DECAY_EPOCHS")
fi

if [[ "$USE_XMOE_ROUTER" == "True" ]]; then
  cmd+=(--use_xmoe_router --xmoe_router_dim "$XMOE_ROUTER_DIM" --xmoe_noise_scale "$XMOE_NOISE_SCALE")
fi

"${cmd[@]}"

echo "Finished Week27 DeepSeek arch config ${CONFIG_ID}"
