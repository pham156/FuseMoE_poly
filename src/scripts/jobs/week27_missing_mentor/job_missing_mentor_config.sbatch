#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J miss_mentor
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_27/missing_mentor/%j_missing_mentor.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_27/missing_mentor/%j_missing_mentor.err

set -eo pipefail

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

cd /home/pham156/MoE/FuseMoE_poly/src/scripts

CONFIG_ID="${CONFIG_ID:-base_zero}"
SEED="${SEED:-32}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-8}"
BALANCE_COEF="${BALANCE_COEF:-0.01}"
ROUTER_ENTROPY_COEF="${ROUTER_ENTROPY_COEF:-0.01}"
RECON_COEF="${RECON_COEF:-0.05}"
EXPERT_ORTH_COEF="${EXPERT_ORTH_COEF:-0.001}"

USE_LEARNED_MISSING="${USE_LEARNED_MISSING:-0}"
USE_RECON="${USE_RECON:-0}"
USE_EXPERT_ORTH="${USE_EXPERT_ORTH:-0}"

OUT_ROOT="/home/pham156/MoE/FuseMoE_poly/out/Week_27/missing_mentor"
RUN_ROOT="/home/pham156/MoE/FuseMoE_poly/run_folder/week27/missing_mentor"
mkdir -p "$OUT_ROOT"

TASK_SPECS=(
  "ihm-48-cxr-notes-ecg-missingInd 2 f1"
  "los-48-cxr-notes-ecg-missingInd 2 f1"
)

echo "===== WEEK27 MISSING-MENTOR CONFIG START ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "config_id: ${CONFIG_ID}"
echo "seed: ${SEED}"
echo "epochs_per_task: ${TOTAL_EPOCHS}"
echo "router: permod"
echo "gating: softmax"
echo "experts: 4"
echo "top_k: 2"
echo "balance_loss_coef: ${BALANCE_COEF}"
echo "router_entropy_coef: ${ROUTER_ENTROPY_COEF}"
echo "use_learned_missing_embeddings: ${USE_LEARNED_MISSING}"
echo "use_missing_modality_recon: ${USE_RECON}"
echo "missing_modality_recon_coef: ${RECON_COEF}"
echo "use_expert_orth: ${USE_EXPERT_ORTH}"
echo "expert_orth_coef: ${EXPERT_ORTH_COEF}"
echo "=============================================="

for spec in "${TASK_SPECS[@]}"; do
  read -r TASK NUM_LABELS PRIMARY_METRIC <<< "$spec"
  OUTPUT_DIR="${RUN_ROOT}/${CONFIG_ID}/${TASK}"

  echo "----- task_start ${TASK} labels=${NUM_LABELS} primary=${PRIMARY_METRIC} -----"

  cmd=(python -u -W ignore main_mimiciv.py
    --num_train_epochs "$TOTAL_EPOCHS"
    --modeltype 'TS_CXR_Text_ECG'
    --kernel_size 1
    --train_batch_size 2
    --eval_batch_size 8
    --seed "$SEED"
    --gradient_accumulation_steps 16
    --num_update_bert_epochs 2
    --bertcount 0
    --ts_learning_rate 0.0004
    --txt_learning_rate 0.00002
    --notes_order 'Last'
    --num_of_notes 5
    --max_length 1024
    --layers 3
    --output_dir "$OUTPUT_DIR"
    --disable_run_folder_save
    --embed_dim 128
    --num_modalities 4
    --model_name "bioLongformer"
    --dataset "mimic"
    --task "$TASK"
    --primary_metric "$PRIMARY_METRIC"
    --file_path '../../data/MIMIC-IV'
    --num_labels "$NUM_LABELS"
    --train_sample_ratio 1.0
    --num_heads 8
    --embed_time 64
    --tt_max 48
    --TS_mixup
    --mixup_level 'batch'
    --fp16
    --irregular_learn_emb_text
    --irregular_learn_emb_ts
    --irregular_learn_emb_cxr
    --irregular_learn_emb_ecg
    --cross_method "moe"
    --gating_function "softmax"
    --num_of_experts 4
    --top_k 2
    --disjoint_top_k 2
    --hidden_size 512
    --use_pt_text_embeddings
    --router_type "permod"
    --reg_ts
    --use_balance_loss
    --balance_loss_coef "$BALANCE_COEF"
    --router_entropy_coef "$ROUTER_ENTROPY_COEF"
    --noisy_gating True
    --shared_experts 0
    --use_bias False
    --use_temp False
    --normalized True
    --router_print_mode concise)

  if [[ "$USE_LEARNED_MISSING" == "1" ]]; then
    cmd+=(--use_learned_missing_embeddings)
  fi

  if [[ "$USE_RECON" == "1" ]]; then
    cmd+=(--use_missing_modality_recon
      --missing_modality_recon_coef "$RECON_COEF"
      --missing_modality_recon_targets "cxr,ecg,text"
      --missing_modality_recon_hidden 256)
  fi

  if [[ "$USE_EXPERT_ORTH" == "1" ]]; then
    cmd+=(--expert_orth_coef "$EXPERT_ORTH_COEF")
  fi

  "${cmd[@]}"
  echo "----- task_done ${TASK} -----"
done

echo "Finished Week27 missing-mentor config ${CONFIG_ID}"

