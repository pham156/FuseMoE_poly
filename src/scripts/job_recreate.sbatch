#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J moe_inner_%j
#SBATCH -o ../../out/Week_13/moe_%j.out
#SBATCH -e ../../out/Week_13/moe_%j.err

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
# export WANDB_MODE=offline
export WANDB_DISABLED=true

while [[ $# -gt 0 ]]; do
  case $1 in
    # --seed) SEED="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --num_labels) NUM_LABELS="$2"; shift 2 ;;
    --primary_metric) PRIMARY_METRIC="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --gating) GATING="$2"; shift 2 ;;
    --experts) EXPERTS="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --shared_experts) SHARED_EXPERTS="$2"; shift 2 ;;
    --use_temp) USE_TEMP="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

cd /home/pham156/MoE/FuseMoE_poly/src/scripts

# Loop over powers inside the job
for RATIO in 0.02 0.05 0.1 0.2 0.4 0.6 0.8; do
  for ROUTER in "joint"; do
    echo "Running task=$TASK, ratio=$RATIO, gating=$GATING, seed=$SEED, shared_experts=$SHARED_EXPERTS, router_type=$ROUTER, num_experts=$EXPERTS, top_k=$TOP_K, noisy_gating=True, task=$TASK, use_bias=True, use_temp=$USE_TEMP"

    python -W ignore main_mimiciv.py \
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
      --output_dir "../../run_folder/week3/run_mimiciv_poly/TS_CXR_Text" \
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
      --gating_function "$GATING" \
      --num_of_experts "$EXPERTS" \
      --top_k "$TOP_K" \
      --disjoint_top_k "$TOP_K" \
      --hidden_size 512 \
      --use_pt_text_embeddings \
      --router_type "$ROUTER" \
      --reg_ts \
      --use_balance_loss \
      --balance_loss_coef 0.01 \
      --noisy_gating True \
      --shared_experts "$SHARED_EXPERTS" \
      --use_bias True \
      --use_temp "$USE_TEMP"

    echo "Finished gating=$GATING, router_type=$ROUTER, num_experts=$EXPERTS, top_k=$TOP_K, shared_experts=$SHARED_EXPERTS, noisy_gating=True, task=$TASK"
    echo "---------------------------------------------"
  done
done