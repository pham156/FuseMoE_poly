#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J pam_moe_%j
#SBATCH -o ../../out/Week_11/pam_%j.out
#SBATCH -e ../../out/Week_11/pam_%j.err

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_MODE=offline

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --seed) SEED="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --num_labels) NUM_LABELS="$2"; shift 2 ;;
    --primary_metric) PRIMARY_METRIC="$2"; shift 2 ;;
    --experts) EXPERTS="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --normalized) NORMALIZED="$2"; shift 2 ;;
    --shared_experts) SHARED_EXPERTS="$2"; shift 2 ;;
    --router) ROUTER="$2"; shift 2 ;;
    --gating) GATING="$2"; shift 2 ;;
    --poly_power) POLY_POWER="$2"; shift 2 ;;
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

cd /home/pham156/MoE/FuseMoE_poly/src/scripts

# Build the command with conditional poly_power
if [ "$GATING" = "poly" ]; then
  POLY_ARG="--poly_power $POLY_POWER"
else
  POLY_ARG=""
fi

python -W ignore main_mimiciv.py \
  --num_train_epochs 20 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --seed "$SEED" \
  --gradient_accumulation_steps 4 \
  --ts_learning_rate 0.001 \
  --task "$TASK" \
  --primary_metric "$PRIMARY_METRIC" \
  --file_path "$DATA_DIR" \
  --num_labels "$NUM_LABELS" \
  --embed_dim 128 \
  --num_heads 8 \
  --tt_max 100 \
  --cross_method "moe" \
  --gating_function "$GATING" \
  $POLY_ARG \
  --num_of_experts "$EXPERTS" \
  --top_k "$TOP_K" \
  --disjoint_top_k "$TOP_K" \
  --hidden_size 512 \
  --router_type "$ROUTER" \
  --use_balance_loss \
  --balance_loss_coef 0.01 \
  --noisy_gating True \
  --normalized "$NORMALIZED" \
  --use_bias False \
  --shared_experts "$SHARED_EXPERTS" \
  --dataset "pam" \
  --output_dir "../../run_folder/pam_runs"