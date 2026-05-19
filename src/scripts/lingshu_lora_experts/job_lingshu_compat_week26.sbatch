#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 0:45:00
#SBATCH -J lingshu_compat26_%j
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_26/lingshu_compat26_%j.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_26/lingshu_compat26_%j.err

set -eo pipefail

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

TASK=""
NUM_LABELS=""
PRIMARY_METRIC=""
SEED="30"
RATIO="0.025"
EPOCHS="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --num_labels) NUM_LABELS="$2"; shift 2 ;;
    --primary_metric) PRIMARY_METRIC="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --ratio) RATIO="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$TASK" || -z "$NUM_LABELS" || -z "$PRIMARY_METRIC" ]]; then
  echo "Missing required task arguments" >&2
  exit 1
fi

cd /home/pham156/MoE/FuseMoE_poly/src/scripts
mkdir -p /home/pham156/MoE/FuseMoE_poly/out/Week_26

LINGSHU_MODEL_PATH="${LINGSHU_MODEL_PATH:-/scratch/gilbreth/pham156/hf_cache/lingshu-medical-mllm/Lingshu-7B}"
if [[ ! -f "$LINGSHU_MODEL_PATH/config.json" ]]; then
  echo "Missing Lingshu config.json at $LINGSHU_MODEL_PATH" >&2
  exit 1
fi

OUTPUT_DIR="/home/pham156/MoE/FuseMoE_poly/run_folder/week26/lingshu_compat/${TASK}_seed${SEED}/ratio_${RATIO}"

echo "===== WEEK26 LINGSHU COMPATIBILITY RUN START ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "lingshu_model_path: $LINGSHU_MODEL_PATH"
echo "task: $TASK"
echo "ratio: $RATIO"
echo "seed: $SEED"
echo "epochs: $EPOCHS"
echo "num_labels: $NUM_LABELS"
echo "primary_metric: $PRIMARY_METRIC"
echo "output_dir: $OUTPUT_DIR"
echo "=================================================="

python -u -W ignore main_mimiciv_lingshu.py \
  --num_train_epochs "$EPOCHS" \
  --modeltype 'TS_CXR_Text' \
  --kernel_size 1 \
  --train_batch_size 1 \
  --eval_batch_size 4 \
  --seed "$SEED" \
  --gradient_accumulation_steps 16 \
  --num_update_bert_epochs 2 \
  --bertcount 0 \
  --ts_learning_rate 0.0001 \
  --txt_learning_rate 0.00002 \
  --notes_order 'Last' \
  --num_of_notes 5 \
  --max_length 1024 \
  --layers 3 \
  --output_dir "$OUTPUT_DIR" \
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
  --hidden_size 512 \
  --use_pt_text_embeddings \
  --router_type "joint" \
  --reg_ts \
  --noisy_gating True \
  --shared_experts 0 \
  --normalized True \
  --lingshu_model_path "$LINGSHU_MODEL_PATH" \
  --lingshu_freeze_backbone True \
  --lingshu_use_peft_lora True \
  --lingshu_lora_r 8 \
  --lingshu_lora_alpha 16 \
  --lingshu_lora_dropout 0.05 \
  --lingshu_max_ts_tokens 48 \
  --lingshu_pooling mean

echo "Lingshu compatibility run finished task=$TASK seed=$SEED ratio=$RATIO"
