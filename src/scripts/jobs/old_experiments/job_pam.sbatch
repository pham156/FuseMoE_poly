#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J pam_moe_%j
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_21/pam_%j.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_21/pam_%j.err

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

NUM_TRAIN_EPOCHS=8
TRAIN_BATCH_SIZE=2
EVAL_BATCH_SIZE=8
TS_LEARNING_RATE=0.0004
GRAD_ACCUM_STEPS=4
EMBED_DIM=128
NUM_HEADS=8
TT_MAX=100

TRAIN_SUBJECTS=(3 4 5 6 7 8)
VAL_SUBJECTS=(2)
TEST_SUBJECTS=(1)

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --seed_batch) SEED_BATCH="$2"; shift 2 ;;
    --seed_list) SEED_LIST="$2"; shift 2 ;;
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

IFS=',' read -r -a SEEDS <<< "$SEED_LIST"

echo "===== PAM RUN CONFIG ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "seed_batch: $SEED_BATCH"
echo "seed_list: ${SEEDS[*]}"
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
echo "num_train_epochs: $NUM_TRAIN_EPOCHS"
echo "train_batch_size: $TRAIN_BATCH_SIZE"
echo "eval_batch_size: $EVAL_BATCH_SIZE"
echo "gradient_accumulation_steps: $GRAD_ACCUM_STEPS"
echo "ts_learning_rate: $TS_LEARNING_RATE"
echo "embed_dim: $EMBED_DIM"
echo "num_heads: $NUM_HEADS"
echo "tt_max: $TT_MAX"
echo "fp16: True"
echo "data_dir: $DATA_DIR"
echo "output_dir: ../../run_folder/pam_runs"
echo "=========================="

for seed in "${SEEDS[@]}"; do
  SEED=$seed
  echo "Starting PAM run with seed $SEED at $(date)..."
  python -u -W ignore main_mimiciv.py \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --seed "$SEED" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --ts_learning_rate "$TS_LEARNING_RATE" \
    --task "$TASK" \
    --primary_metric "$PRIMARY_METRIC" \
    --pam_train_subjects "${TRAIN_SUBJECTS[@]}" \
    --pam_val_subjects "${VAL_SUBJECTS[@]}" \
    --pam_test_subjects "${TEST_SUBJECTS[@]}" \
    --file_path "$DATA_DIR" \
    --num_labels "$NUM_LABELS" \
    --embed_dim "$EMBED_DIM" \
    --num_heads "$NUM_HEADS" \
    --tt_max "$TT_MAX" \
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
    --fp16 \
    --dataset "pam" \
    --output_dir "../../run_folder/pam_runs"
done

rc=$?
echo "===== PAM RUN END ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "seed_batch: $SEED_BATCH"
echo "last_seed: $SEED"
echo "exit_code: $rc"
echo "======================="
exit $rc
