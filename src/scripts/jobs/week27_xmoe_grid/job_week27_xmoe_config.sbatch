#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J w27_xmoe
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_27/%j_week27_xmoe.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_27/%j_week27_xmoe.err

set -eo pipefail

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

cd /home/pham156/MoE/FuseMoE_poly/src/scripts

CONFIG_ID="${CONFIG_ID:-manual}"
SEED="${SEED:-32}"
GATING_FUNCTION="${GATING_FUNCTION:-softmax}"
POLY_POWER="${POLY_POWER:-4}"
USE_XMOE_ROUTER="${USE_XMOE_ROUTER:-False}"
ROUTER="${ROUTER:-joint}"
MODELTYPE="${MODELTYPE:-TS_CXR_Text}"
NUM_MODALITIES="${NUM_MODALITIES:-3}"
SHARED_EXPERTS="${SHARED_EXPERTS:-1}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-16}"
NUM_EXPERTS="${NUM_EXPERTS:-4}"
TOP_K="${TOP_K:-2}"
DISJOINT_TOP_K="${DISJOINT_TOP_K:-2}"
BALANCE_COEF="${BALANCE_COEF:-0.01}"
XMOE_ROUTER_DIM="${XMOE_ROUTER_DIM:-128}"
XMOE_NOISE_SCALE="${XMOE_NOISE_SCALE:-1.0}"
ROUTER_PRINT_MODE="${ROUTER_PRINT_MODE:-concise}"

TASK_SPECS=(
  "ihm-48-cxr-notes-ecg 2 f1"
  "los-48-cxr-notes-ecg 2 f1"
  "pheno-all-cxr-notes-ecg 25 macro_f1"
)

echo "===== WEEK27 XMOE GRID CONFIG START ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "config_id: ${CONFIG_ID}"
echo "seed: ${SEED}"
echo "gating: ${GATING_FUNCTION}"
if [[ "$GATING_FUNCTION" == "poly" ]]; then
  echo "poly_power: ${POLY_POWER}"
fi
echo "use_xmoe_router: ${USE_XMOE_ROUTER}"
echo "router: ${ROUTER}"
echo "modeltype: ${MODELTYPE}"
echo "num_modalities: ${NUM_MODALITIES}"
echo "shared_experts: ${SHARED_EXPERTS}"
echo "experts: ${NUM_EXPERTS}"
echo "top_k: ${TOP_K}"
echo "epochs_per_task: ${TOTAL_EPOCHS}"
echo "balance_loss_coef: ${BALANCE_COEF}"
echo "router_print_mode: ${ROUTER_PRINT_MODE}"
echo "========================================="

for spec in "${TASK_SPECS[@]}"; do
  read -r TASK NUM_LABELS PRIMARY_METRIC <<< "$spec"
  OUTPUT_DIR="/home/pham156/MoE/FuseMoE_poly/run_folder/week27/xmoe_grid/${CONFIG_ID}/${TASK}"

  echo "----- task_start ${TASK} labels=${NUM_LABELS} primary=${PRIMARY_METRIC} -----"

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
    --irregular_learn_emb_ecg \
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
    --noisy_gating True \
    --shared_experts "$SHARED_EXPERTS" \
    --use_bias False \
    --use_temp False \
    --normalized True \
    --router_temperature 1.0 \
    --router_print_mode "$ROUTER_PRINT_MODE")

  if [[ "$USE_XMOE_ROUTER" == "True" || "$USE_XMOE_ROUTER" == "true" || "$USE_XMOE_ROUTER" == "1" ]]; then
    cmd+=(--use_xmoe_router --xmoe_router_dim "$XMOE_ROUTER_DIM" --xmoe_noise_scale "$XMOE_NOISE_SCALE")
  fi

  if [[ "$GATING_FUNCTION" == "poly" ]]; then
    cmd+=(--poly_power "$POLY_POWER")
  fi

  "${cmd[@]}"
  echo "----- task_done ${TASK} -----"
done

echo "Finished Week27 XMOE grid config ${CONFIG_ID}"
