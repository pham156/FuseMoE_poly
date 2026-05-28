#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 4:00:00
#SBATCH -J lingshu_focus
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_27/lingshu_focus/%j_lingshu_focus.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_27/lingshu_focus/%j_lingshu_focus.err

set -eo pipefail

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

TASK="${TASK:-ihm-48-cxr-notes-ecg}"
NUM_LABELS="${NUM_LABELS:-2}"
PRIMARY_METRIC="${PRIMARY_METRIC:-f1}"
SEED="${SEED:-32}"
RATIO="${RATIO:-0.025}"
EPOCHS="${EPOCHS:-1}"
ARCH="${ARCH:-organ_lora_moe}"
ROUTER="${ROUTER:-linear}"
MODELTYPE="${MODELTYPE:-TS_CXR_Text_ECG}"
NUM_MODALITIES="${NUM_MODALITIES:-4}"
TRAIN_BS="${TRAIN_BS:-1}"
EVAL_BS="${EVAL_BS:-4}"
LR="${LR:-0.0001}"
OUT_TAG="${OUT_TAG:-smoke}"

cd /home/pham156/MoE/FuseMoE_poly/src/scripts
mkdir -p /home/pham156/MoE/FuseMoE_poly/out/Week_27/lingshu_focus

LINGSHU_MODEL_PATH="${LINGSHU_MODEL_PATH:-/scratch/gilbreth/pham156/hf_cache/lingshu-medical-mllm/Lingshu-7B}"
if [[ -d "$LINGSHU_MODEL_PATH" && -z "$(find "$LINGSHU_MODEL_PATH" -maxdepth 1 -type f -name config.json -print -quit)" ]]; then
  SNAPSHOT="$(find "$LINGSHU_MODEL_PATH" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  if [[ -n "$SNAPSHOT" ]]; then
    LINGSHU_MODEL_PATH="$SNAPSHOT"
  fi
fi

if [[ ! -f "$LINGSHU_MODEL_PATH/config.json" ]]; then
  echo "Missing Lingshu config.json at $LINGSHU_MODEL_PATH" >&2
  exit 1
fi

OUTPUT_DIR="/home/pham156/MoE/FuseMoE_poly/run_folder/week27/lingshu_focus/${OUT_TAG}/${TASK}_arch${ARCH}_seed${SEED}_ratio${RATIO}"

echo "===== WEEK27 LINGSHU FOCUS START ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "task: $TASK"
echo "num_labels: $NUM_LABELS"
echo "primary_metric: $PRIMARY_METRIC"
echo "architecture: $ARCH"
echo "router: $ROUTER"
echo "modeltype: $MODELTYPE"
echo "ratio: $RATIO"
echo "seed: $SEED"
echo "epochs: $EPOCHS"
echo "lingshu_model_path: $LINGSHU_MODEL_PATH"
echo "output_dir: $OUTPUT_DIR"
echo "======================================"

python -u -W ignore main_mimiciv_lingshu.py \
  --lingshu_architecture "$ARCH" \
  --lingshu_moe_router "$ROUTER" \
  --lingshu_moe_top_k 2 \
  --num_train_epochs "$EPOCHS" \
  --modeltype "$MODELTYPE" \
  --kernel_size 1 \
  --train_batch_size "$TRAIN_BS" \
  --eval_batch_size "$EVAL_BS" \
  --seed "$SEED" \
  --gradient_accumulation_steps 16 \
  --num_update_bert_epochs 2 \
  --bertcount 0 \
  --ts_learning_rate "$LR" \
  --txt_learning_rate 0.00002 \
  --notes_order 'Last' \
  --num_of_notes 5 \
  --max_length 1024 \
  --layers 3 \
  --output_dir "$OUTPUT_DIR" \
  --embed_dim 128 \
  --num_modalities "$NUM_MODALITIES" \
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
  --balance_loss_coef 0.01 \
  --noisy_gating True \
  --shared_experts 0 \
  --use_bias False \
  --use_temp False \
  --normalized True \
  --lingshu_model_path "$LINGSHU_MODEL_PATH" \
  --lingshu_freeze_backbone True \
  --lingshu_use_peft_lora True \
  --lingshu_lora_r 8 \
  --lingshu_lora_alpha 16 \
  --lingshu_lora_dropout 0.05 \
  --lingshu_max_ts_tokens 48 \
  --lingshu_pooling mean \
  --disable_run_folder_save

echo "Lingshu focus run finished."
