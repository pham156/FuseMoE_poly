#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 2:00:00
#SBATCH -J lingshu26_%j
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_26/lingshu26_%j.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_26/lingshu26_%j.err

set -eo pipefail

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

SEED=""
RATIOS_OVERRIDE=""
EPOCHS="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --ratios) RATIOS_OVERRIDE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$SEED" ]]; then
  echo "Missing required --seed argument" >&2
  exit 1
fi

cd /home/pham156/MoE/FuseMoE_poly/src/scripts
mkdir -p /home/pham156/MoE/FuseMoE_poly/out/Week_26

LINGSHU_MODEL_PATH="${LINGSHU_MODEL_PATH:-/scratch/gilbreth/pham156/hf_cache/lingshu-medical-mllm/Lingshu-7B}"
if [[ -d "$LINGSHU_MODEL_PATH" && -z "$(find "$LINGSHU_MODEL_PATH" -maxdepth 1 -type f -name config.json -print -quit)" ]]; then
  SNAPSHOT="$(find "$LINGSHU_MODEL_PATH" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
  if [[ -n "$SNAPSHOT" ]]; then
    LINGSHU_MODEL_PATH="$SNAPSHOT"
  fi
fi

if [[ ! -f "$LINGSHU_MODEL_PATH/config.json" ]]; then
  echo "Missing Lingshu config.json at $LINGSHU_MODEL_PATH" >&2
  echo "Set LINGSHU_MODEL_PATH to a local Lingshu snapshot before submitting this job." >&2
  exit 1
fi

if [[ -n "$RATIOS_OVERRIDE" ]]; then
  read -r -a RATIOS <<< "$RATIOS_OVERRIDE"
else
  RATIOS=(0.05 0.1 0.4 0.8)
fi

TASK="ihm-48-cxr-notes-ecg"

for RATIO in "${RATIOS[@]}"; do
  OUTPUT_DIR="/home/pham156/MoE/FuseMoE_poly/run_folder/week26/lingshu_pseudotoken/${TASK}_seed${SEED}/ratio_${RATIO}"

  echo "===== WEEK26 LINGSHU PSEUDO-TOKEN RUN START ====="
  echo "job_id: ${SLURM_JOB_ID:-N/A}"
  echo "lingshu_model_path: $LINGSHU_MODEL_PATH"
  echo "task: $TASK"
  echo "ratio: $RATIO"
  echo "seed: $SEED"
  echo "epochs: $EPOCHS"
  echo "output_dir: $OUTPUT_DIR"
  echo "================================================="

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

  echo "Finished Lingshu pseudo-token run task=$TASK ratio=$RATIO seed=$SEED"
  echo "----------------------------------------------------------------"
done
