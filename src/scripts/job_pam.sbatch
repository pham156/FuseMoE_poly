#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J pam_moe_%j
#SBATCH -o ../../out/Week_15/pam_%j.out
#SBATCH -e ../../out/Week_15/pam_%j.err

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_MODE=offline

TRAIN_SUBJECTS=(3 4 5 6 7 8)
VAL_SUBJECTS=(2)
TEST_SUBJECTS=(1)

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

echo "===== PAM RUN CONFIG ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "seed: $SEED"
echo "task: $TASK"
echo "num_labels: $NUM_LABELS"
echo "primary_metric: $PRIMARY_METRIC"
echo "pam_train_subjects: ${TRAIN_SUBJECTS[*]}"
echo "pam_val_subjects: ${VAL_SUBJECTS[*]}"
echo "pam_test_subjects: ${TEST_SUBJECTS[*]}"
echo "cross_method: moe"
echo "router_type: $ROUTER"
echo "gating_function: $GATING"
echo "poly_power: ${POLY_POWER:-N/A}"
echo "num_of_experts: $EXPERTS"
echo "top_k: $TOP_K"
echo "normalized: $NORMALIZED"
echo "shared_experts: $SHARED_EXPERTS"
echo "use_balance_loss: True"
echo "balance_loss_coef: 0.01"
echo "num_train_epochs: 30"
echo "train_batch_size: 16"
echo "eval_batch_size: 32"
echo "gradient_accumulation_steps: 4"
echo "ts_learning_rate: 0.0001"
echo "embed_dim: 128"
echo "num_heads: 8"
echo "tt_max: 100"
echo "data_dir: $DATA_DIR"
echo "output_dir: ../../run_folder/pam_runs"
echo "=========================="

for seed in 22 32 52 62 72 82 92 102 112 122; do
  SEED=$seed
  echo "Starting PAM run with seed $SEED..."
  python -W ignore main_mimiciv.py \
    --num_train_epochs 8 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --seed "$SEED" \
    --gradient_accumulation_steps 4 \
    --ts_learning_rate 0.0001 \
    --task "$TASK" \
    --primary_metric "$PRIMARY_METRIC" \
    --pam_train_subjects "${TRAIN_SUBJECTS[@]}" \
    --pam_val_subjects "${VAL_SUBJECTS[@]}" \
    --pam_test_subjects "${TEST_SUBJECTS[@]}" \
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
done

rc=$?
echo "===== PAM RUN END ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "seed: $SEED"
echo "exit_code: $rc"
echo "======================="
exit $rc
