#!/bin/bash
#SBATCH -A ai-forge
#SBATCH -p a100-80gb
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J paper_cfg26
#SBATCH -o /home/pham156/MoE/FuseMoE_poly/out/Week_26/%j_paper_cfg26.out
#SBATCH -e /home/pham156/MoE/FuseMoE_poly/out/Week_26/%j_paper_cfg26.err

set -eo pipefail

module load anaconda
conda activate /scratch/gilbreth/pham156/conda_envs/fusemoe
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

cd /home/pham156/MoE/FuseMoE_poly/src/scripts

TASK="${TASK:-ihm-48-cxr-notes-ecg}"
MODELTYPE="${MODELTYPE:-TS_CXR_Text}"
NUM_MODALITIES="${NUM_MODALITIES:-3}"
NUM_LABELS="${NUM_LABELS:-2}"
PRIMARY_METRIC="${PRIMARY_METRIC:-f1}"
SEED="${SEED:-30}"
RATIO="${RATIO:-1.0}"
GATING_FUNCTION="${GATING_FUNCTION:-softmax}"
POLY_POWER="${POLY_POWER:-2}"
BALANCE_COEF="${BALANCE_COEF:-0.01}"
NOISY_GATING="${NOISY_GATING:-True}"
ROUTER="${ROUTER:-joint}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-8}"
NUM_EXPERTS="${NUM_EXPERTS:-16}"
TOP_K="${TOP_K:-4}"
DISJOINT_TOP_K="${DISJOINT_TOP_K:-2}"
SHARED_EXPERTS="${SHARED_EXPERTS:-0}"
USE_XMOE_ROUTER="${USE_XMOE_ROUTER:-False}"
XMOE_ROUTER_DIM="${XMOE_ROUTER_DIM:-128}"
XMOE_NOISE_SCALE="${XMOE_NOISE_SCALE:-1.0}"
ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-1.0}"
LOG_ROUTER_DIAGNOSTICS="${LOG_ROUTER_DIAGNOSTICS:-False}"
ROUTER_DIAGNOSTICS_PATH="${ROUTER_DIAGNOSTICS_PATH:-}"
USE_TEMP="${USE_TEMP:-False}"
USE_SEMANTIC_EXPERT_PROFILES="${USE_SEMANTIC_EXPERT_PROFILES:-False}"
SEMANTIC_PROFILE_SOURCE="${SEMANTIC_PROFILE_SOURCE:-patient}"
SEMANTIC_PROFILE_FUSION="${SEMANTIC_PROFILE_FUSION:-add}"
SEMANTIC_PROFILE_LAYERS="${SEMANTIC_PROFILE_LAYERS:-all}"
SEMANTIC_PROFILE_SCALE="${SEMANTIC_PROFILE_SCALE:-1.0}"
SEMANTIC_PROFILE_NOTE_POOLING="${SEMANTIC_PROFILE_NOTE_POOLING:-max}"

SAFE_RATIO="${RATIO/./p}"
SAFE_BAL="${BALANCE_COEF/./p}"
SAFE_GATE="${GATING_FUNCTION//[^A-Za-z0-9_]/_}"
SAFE_XMOE="$(echo "$USE_XMOE_ROUTER" | tr '[:upper:]' '[:lower:]')"

OUTPUT_DIR="/home/pham156/MoE/FuseMoE_poly/run_folder/week26/paper_config/${SAFE_GATE}_${TASK}_${ROUTER}_e${NUM_EXPERTS}_k${TOP_K}_seed${SEED}_ratio${SAFE_RATIO}_bal${SAFE_BAL}_ep${TOTAL_EPOCHS}_shared${SHARED_EXPERTS}_xmoe${SAFE_XMOE}_dim${XMOE_ROUTER_DIM}"

echo "===== WEEK26 PAPER-CONFIG OLD FUSEMOE START ====="
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "task: ${TASK}"
echo "modeltype: ${MODELTYPE}"
echo "num_modalities: ${NUM_MODALITIES}"
echo "seed: ${SEED}"
echo "ratio: ${RATIO}"
echo "router: ${ROUTER}"
echo "gating: ${GATING_FUNCTION}"
if [[ "$GATING_FUNCTION" == "poly" ]]; then
  echo "poly_power: ${POLY_POWER}"
fi
echo "experts: ${NUM_EXPERTS}"
echo "top_k: ${TOP_K}"
echo "disjoint_top_k: ${DISJOINT_TOP_K}"
echo "layers: 3"
echo "epochs: ${TOTAL_EPOCHS}"
echo "hidden_size: 512"
echo "balance_loss_coef: ${BALANCE_COEF}"
echo "noisy_gating: ${NOISY_GATING}"
echo "shared_experts: ${SHARED_EXPERTS}"
echo "use_xmoe_router: ${USE_XMOE_ROUTER}"
echo "xmoe_router_dim: ${XMOE_ROUTER_DIM}"
echo "xmoe_noise_scale: ${XMOE_NOISE_SCALE}"
echo "use_temp: ${USE_TEMP}"
echo "router_temperature: ${ROUTER_TEMPERATURE}"
echo "use_semantic_expert_profiles: ${USE_SEMANTIC_EXPERT_PROFILES}"
echo "semantic_profile_source: ${SEMANTIC_PROFILE_SOURCE}"
echo "semantic_profile_fusion: ${SEMANTIC_PROFILE_FUSION}"
echo "semantic_profile_layers: ${SEMANTIC_PROFILE_LAYERS}"
echo "semantic_profile_scale: ${SEMANTIC_PROFILE_SCALE}"
echo "semantic_profile_note_pooling: ${SEMANTIC_PROFILE_NOTE_POOLING}"
echo "log_router_diagnostics: ${LOG_ROUTER_DIAGNOSTICS}"
echo "router_diagnostics_path: ${ROUTER_DIAGNOSTICS_PATH:-default}"
if [[ "$LOG_ROUTER_DIAGNOSTICS" == "True" || "$LOG_ROUTER_DIAGNOSTICS" == "true" || "$LOG_ROUTER_DIAGNOSTICS" == "1" ]]; then
  echo "router diagnostics CSV: enabled"
else
  echo "router diagnostics CSV: disabled"
fi
echo "output_dir: ${OUTPUT_DIR}"
echo "================================================="

cmd=(python -u -W ignore main_mimiciv.py \
  --num_train_epochs "$TOTAL_EPOCHS" \
  --modeltype "$MODELTYPE" \
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
  --disable_run_folder_save \
  --embed_dim 128 \
  --num_modalities "$NUM_MODALITIES" \
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
  --gating_function "$GATING_FUNCTION" \
  --num_of_experts "$NUM_EXPERTS" \
  --top_k "$TOP_K" \
  --disjoint_top_k "$DISJOINT_TOP_K" \
  --hidden_size 512 \
  --use_pt_text_embeddings \
  --router_type "$ROUTER" \
  --reg_ts \
  --use_balance_loss \
  --balance_loss_coef "$BALANCE_COEF" \
  --noisy_gating "$NOISY_GATING" \
  --shared_experts "$SHARED_EXPERTS" \
  --use_bias False \
  --use_temp "$USE_TEMP" \
  --normalized True \
  --router_temperature "$ROUTER_TEMPERATURE")

if [[ "$USE_XMOE_ROUTER" == "True" || "$USE_XMOE_ROUTER" == "true" || "$USE_XMOE_ROUTER" == "1" ]]; then
  cmd+=(--use_xmoe_router --xmoe_router_dim "$XMOE_ROUTER_DIM" --xmoe_noise_scale "$XMOE_NOISE_SCALE")
fi

if [[ "$GATING_FUNCTION" == "poly" ]]; then
  cmd+=(--poly_power "$POLY_POWER")
fi

if [[ "$LOG_ROUTER_DIAGNOSTICS" == "True" || "$LOG_ROUTER_DIAGNOSTICS" == "true" || "$LOG_ROUTER_DIAGNOSTICS" == "1" ]]; then
  cmd+=(--log_router_diagnostics --router_diagnostics_layers all)
  if [[ -n "$ROUTER_DIAGNOSTICS_PATH" ]]; then
    cmd+=(--router_diagnostics_path "$ROUTER_DIAGNOSTICS_PATH")
  fi
fi

if [[ "$USE_SEMANTIC_EXPERT_PROFILES" == "True" || "$USE_SEMANTIC_EXPERT_PROFILES" == "true" || "$USE_SEMANTIC_EXPERT_PROFILES" == "1" ]]; then
  cmd+=(--use_semantic_expert_profiles \
    --semantic_profile_set icu_organ_system \
    --semantic_profile_source "$SEMANTIC_PROFILE_SOURCE" \
    --semantic_profile_fusion "$SEMANTIC_PROFILE_FUSION" \
    --semantic_profile_layers "$SEMANTIC_PROFILE_LAYERS" \
    --semantic_profile_scale "$SEMANTIC_PROFILE_SCALE" \
    --semantic_profile_note_pooling "$SEMANTIC_PROFILE_NOTE_POOLING")
fi

"${cmd[@]}"

echo "Finished Week26 paper-config old FuseMoE"
