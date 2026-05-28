#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH -t 4:00:00
#SBATCH -J lingshu_cmp
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_27/lingshu_compare/%j_%x.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_27/lingshu_compare/%j_%x.err

set -eo pipefail

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

ARCH="${ARCH:?ARCH is required: lingshu or base_shared}"
SEED="${SEED:?SEED is required}"
RATIO="${RATIO:-1.0}"
EPOCHS="${EPOCHS:-16}"
MODELTYPE="${MODELTYPE:-TS_CXR_Text}"
NUM_MODALITIES="${NUM_MODALITIES:-3}"
TRAIN_BS="${TRAIN_BS:-1}"
EVAL_BS="${EVAL_BS:-4}"

if [[ -n "${TASK_SPEC:-}" ]]; then
  TASK_SPECS=("$TASK_SPEC")
else
  TASK_SPECS=(
    "ihm-48-cxr-notes-ecg 2 f1"
    "los-48-cxr-notes-ecg 2 f1"
    "pheno-all-cxr-notes-ecg 25 macro_f1"
  )
fi

cd /home/pham156/MoE/FuseMoE_poly/src/scripts
mkdir -p /home/pham156/MoE/FuseMoE_poly/out/Week_27/lingshu_compare

echo "===== WEEK27 LINGSHU COMPARE TASK LOOP START ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "arch: $ARCH"
echo "seed: $SEED"
echo "ratio: $RATIO"
echo "epochs: $EPOCHS"
echo "modeltype: $MODELTYPE"
echo "num_modalities: $NUM_MODALITIES"
echo "tasks: ${TASK_SPECS[*]}"
echo "=================================================="

for task_spec in "${TASK_SPECS[@]}"; do
  read -r TASK NUM_LABELS PRIMARY_METRIC <<< "$task_spec"
  OUTPUT_DIR="/home/pham156/MoE/FuseMoE_poly/run_folder/week27/lingshu_compare/${ARCH}/${TASK}_seed${SEED}_ratio${RATIO}_ep${EPOCHS}"

  echo "----- TASK START arch=$ARCH task=$TASK seed=$SEED -----"

  COMMON_ARGS=(
    --num_train_epochs "$EPOCHS"
    --modeltype "$MODELTYPE"
    --kernel_size 1
    --train_batch_size "$TRAIN_BS"
    --eval_batch_size "$EVAL_BS"
    --seed "$SEED"
    --gradient_accumulation_steps 16
    --num_update_bert_epochs 2
    --bertcount 0
    --ts_learning_rate 0.0001
    --txt_learning_rate 0.00002
    --notes_order Last
    --num_of_notes 5
    --max_length 1024
    --layers 3
    --output_dir "$OUTPUT_DIR"
    --embed_dim 128
    --num_modalities "$NUM_MODALITIES"
    --model_name bioLongformer
    --dataset mimic
    --task "$TASK"
    --primary_metric "$PRIMARY_METRIC"
    --file_path ../../data/MIMIC-IV
    --num_labels "$NUM_LABELS"
    --train_sample_ratio "$RATIO"
    --num_heads 8
    --embed_time 64
    --tt_max 48
    --TS_mixup
    --mixup_level batch
    --fp16
    --irregular_learn_emb_text
    --irregular_learn_emb_ts
    --irregular_learn_emb_cxr
    --cross_method moe
    --gating_function softmax
    --num_of_experts 4
    --top_k 2
    --disjoint_top_k 2
    --hidden_size 512
    --use_pt_text_embeddings
    --router_type joint
    --reg_ts
    --use_balance_loss
    --balance_loss_coef 0.01
    --noisy_gating True
    --shared_experts 1
    --use_bias False
    --use_temp False
    --normalized True
    --disable_run_folder_save
  )

  if [[ "$ARCH" == "lingshu" ]]; then
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
    python -u -W ignore main_mimiciv_lingshu.py \
      "${COMMON_ARGS[@]}" \
      --lingshu_architecture organ_lora_moe \
      --lingshu_moe_router linear \
      --lingshu_moe_top_k 2 \
      --lingshu_model_path "$LINGSHU_MODEL_PATH" \
      --lingshu_freeze_backbone True \
      --lingshu_use_peft_lora True \
      --lingshu_lora_r 8 \
      --lingshu_lora_alpha 16 \
      --lingshu_lora_dropout 0.05 \
      --lingshu_max_ts_tokens 48 \
      --lingshu_pooling mean
  elif [[ "$ARCH" == "base_shared" ]]; then
    python -u -W ignore main_mimiciv.py \
      "${COMMON_ARGS[@]}" \
      --router_print_mode concise
  else
    echo "Unknown ARCH=$ARCH. Use lingshu or base_shared." >&2
    exit 1
  fi

  echo "----- TASK DONE arch=$ARCH task=$TASK seed=$SEED -----"
done

echo "Finished task loop arch=$ARCH seed=$SEED ratio=$RATIO epochs=$EPOCHS"
