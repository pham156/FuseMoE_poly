#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J sem_diag26_%j
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_26/sem_diag26_%j.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_26/sem_diag26_%j.err

# Avoid set -u because Conda hooks can read unset env vars.
set -eo pipefail

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

TASK="ihm-48-cxr-notes-ecg"
NUM_LABELS="2"
PRIMARY_METRIC="f1"
ROUTER="joint"
EXPERTS="4"
TOP_K="2"
DISJOINT_TOP_K="2"
SEMANTIC_PROFILE_SCALE=""
SEMANTIC_PROFILE_FUSION=""
CONFIG_NAME=""
SEEDS_OVERRIDE=""
RATIOS_OVERRIDE=""
EPOCHS="8"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --semantic_profile_scale) SEMANTIC_PROFILE_SCALE="$2"; shift 2 ;;
    --semantic_profile_fusion) SEMANTIC_PROFILE_FUSION="$2"; shift 2 ;;
    --config_name) CONFIG_NAME="$2"; shift 2 ;;
    --seeds) SEEDS_OVERRIDE="$2"; shift 2 ;;
    --ratios) RATIOS_OVERRIDE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$SEMANTIC_PROFILE_SCALE" || -z "$SEMANTIC_PROFILE_FUSION" || -z "$CONFIG_NAME" ]]; then
  echo "Missing required semantic diagnostic arguments" >&2
  exit 1
fi

cd /home/pham156/MoE/FuseMoE_poly/src/scripts
mkdir -p /home/pham156/MoE/FuseMoE_poly/out/Week_26

if [[ -n "$SEEDS_OVERRIDE" ]]; then
  read -r -a SEEDS <<< "$SEEDS_OVERRIDE"
else
  SEEDS=(1 20 30)
fi

if [[ -n "$RATIOS_OVERRIDE" ]]; then
  read -r -a RATIOS <<< "$RATIOS_OVERRIDE"
else
  RATIOS=(0.05 0.1 0.4 0.8)
fi

for SEED in "${SEEDS[@]}"; do
  OUTPUT_DIR_BASE="/home/pham156/MoE/FuseMoE_poly/run_folder/week26/semantic_router_diagnostic/${CONFIG_NAME}_${TASK}_seed${SEED}"

  for RATIO in "${RATIOS[@]}"; do
    OUTPUT_DIR="$OUTPUT_DIR_BASE/ratio_$RATIO"

    echo "===== WEEK26 SEMANTIC ROUTER DIAGNOSTIC START ====="
    echo "job_id: ${SLURM_JOB_ID:-N/A}"
    echo "task: $TASK"
    echo "ratio: $RATIO"
    echo "seed: $SEED"
    echo "router_type: $ROUTER"
    echo "gating_function: softmax"
    echo "num_of_experts: $EXPERTS"
    echo "top_k: $TOP_K"
    echo "disjoint_top_k: $DISJOINT_TOP_K"
    echo "use_instruction_router: False"
    echo "use_semantic_expert_profiles: True"
    echo "semantic_profile_set: icu_organ_system"
    echo "semantic_profile_source: patient"
    echo "semantic_profile_fusion: $SEMANTIC_PROFILE_FUSION"
    echo "semantic_profile_scale: $SEMANTIC_PROFILE_SCALE"
    echo "output_dir: $OUTPUT_DIR"
    echo "==================================================="

    python -u -W ignore main_mimiciv.py \
      --num_train_epochs "$EPOCHS" \
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
      --normalized True \
      --use_semantic_expert_profiles \
      --semantic_profile_set icu_organ_system \
      --semantic_profile_source patient \
      --semantic_profile_fusion "$SEMANTIC_PROFILE_FUSION" \
      --semantic_profile_scale "$SEMANTIC_PROFILE_SCALE"

    echo "Finished semantic diagnostic run task=$TASK ratio=$RATIO seed=$SEED fusion=$SEMANTIC_PROFILE_FUSION scale=$SEMANTIC_PROFILE_SCALE"
    echo "----------------------------------------------------------------"
  done
done
