#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J softmax4_%j
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_22/softmax4_%j.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_22/softmax4_%j.err

# Do not use `set -u` here: Conda's activation/deactivation hooks can read
# unset environment variables and crash before the experiment starts.
set -eo pipefail

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

TASK=""
NUM_LABELS=""
PRIMARY_METRIC=""
SEED=""
ROUTER=""
EXPERTS=""
TOP_K=""
DISJOINT_TOP_K=""
CONFIG_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --num_labels) NUM_LABELS="$2"; shift 2 ;;
    --primary_metric) PRIMARY_METRIC="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --router) ROUTER="$2"; shift 2 ;;
    --experts) EXPERTS="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --disjoint_top_k) DISJOINT_TOP_K="$2"; shift 2 ;;
    --config_name) CONFIG_NAME="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$TASK" || -z "$NUM_LABELS" || -z "$PRIMARY_METRIC" || -z "$SEED" || -z "$ROUTER" || -z "$EXPERTS" || -z "$TOP_K" || -z "$DISJOINT_TOP_K" ]]; then
  echo "Missing required arguments" >&2
  exit 1
fi

cd /home/pham156/MoE/FuseMoE_poly/src/scripts
mkdir -p /home/pham156/MoE/FuseMoE_poly/out/Week_22

RATIOS=(0.025 0.05 0.1 0.2 0.4 0.6 0.8 1.0)
OUTPUT_DIR_BASE="/home/pham156/MoE/FuseMoE_poly/run_folder/week22/reproduce_softmax4_sample_efficiency/${CONFIG_NAME:-softmax}_${TASK}_seed${SEED}"

for RATIO in "${RATIOS[@]}"; do
  OUTPUT_DIR="$OUTPUT_DIR_BASE/ratio_$RATIO"
  echo "===== SOFTMAX SAMPLE EFFICIENCY RUN START ====="
  echo "job_id: ${SLURM_JOB_ID:-N/A}"
  echo "task: $TASK"
  echo "ratio: $RATIO"
  echo "seed: $SEED"
  echo "router_type: $ROUTER"
  echo "gating_function: softmax"
  echo "num_of_experts: $EXPERTS"
  echo "top_k: $TOP_K"
  echo "disjoint_top_k: $DISJOINT_TOP_K"
  echo "shared_experts: 0"
  echo "normalized: True"
  echo "output_dir: $OUTPUT_DIR"
  echo "==============================================="

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
    --balance_loss_coef 0.01 \
    --noisy_gating True \
    --shared_experts 0 \
    --use_bias False \
    --use_temp False \
    --normalized True

  echo "Finished task=$TASK ratio=$RATIO seed=$SEED router=$ROUTER experts=$EXPERTS top_k=$TOP_K"
  echo "------------------------------------------------"
done
