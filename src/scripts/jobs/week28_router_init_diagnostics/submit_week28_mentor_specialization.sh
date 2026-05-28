#!/bin/bash
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/job_week28_mentor_config.sbatch"
OUT_DIR="/home/pham156/MoE/FuseMoE_poly/out/Week_28/mentor_specialization"
MANIFEST="${OUT_DIR}/week28_mentor_specialization_manifest.tsv"
mkdir -p "$OUT_DIR"
printf "job_id\tconfig_id\ttask\tseed\tsemantic_scale\tshared_weight\torth\tvar\tz\tsemantic_only\n" > "$MANIFEST"
TASKS=("ihm-48-cxr-notes-ecg" "los-48-cxr-notes-ecg")
METRICS=("f1" "f1")
LABELS=("2" "2")
SEEDS=("32" "42")
# tag sem_scale shared orth var z sem_only
CONFIGS=(
  "base 0.0 0.0 0.0 0.0 0.0 False"
  "sem01 0.1 0.0 0.0 0.0 0.0 False"
  "sem03 0.3 0.0 0.0 0.0 0.0 False"
  "sem01sh 0.1 1.0 0.0 0.0 0.0 False"
  "sem01reg 0.1 1.0 0.001 0.001 0.001 False"
  "semonly 1.0 0.0 0.0 0.0 0.0 True"
)
for task_idx in "${!TASKS[@]}"; do
  task="${TASKS[$task_idx]}"; metric="${METRICS[$task_idx]}"; labels="${LABELS[$task_idx]}"
  short_task="${task%%-*}"
  for seed in "${SEEDS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
      read -r tag sem shared orth var z semonly <<< "$cfg"
      config_id="w28_mentor_${short_task}_s${seed}_${tag}"
      enable_shared="False"; [[ "$shared" != "0.0" ]] && enable_shared="True"
      use_sem="False"; [[ "$sem" != "0.0" || "$semonly" == "True" ]] && use_sem="True"
      submit_output=$(CONFIG_ID="$config_id" TASK="$task" NUM_LABELS="$labels" PRIMARY_METRIC="$metric" SEED="$seed" ROUTER="permod" SHARED_EXPERTS="0" ARCH_VARIANT="random_low_noise_decay" MODELTYPE="TS_CXR_Text" NUM_MODALITIES="3" TOTAL_EPOCHS="16" NUM_EXPERTS="4" TOP_K="2" BALANCE_COEF="0.01" ROUTER_PRINT_MODE="concise" LOG_EXPERT_OUTPUT_DIAGNOSTICS="True" USE_SEMANTIC_LOGIT_BIAS="$use_sem" SEMANTIC_BIAS_SCALE="$sem" SEMANTIC_PROJECT_DIM="128" SEMANTIC_ONLY_ROUTER="$semonly" SEMANTIC_PROFILE_EMBEDDING_SOURCE="random" ENABLE_SHARED_EXPERT="$enable_shared" SHARED_EXPERT_WEIGHT="$shared" ORTHOGONAL_LOSS_WEIGHT="$orth" ROUTER_VARIANCE_LOSS_WEIGHT="$var" Z_LOSS_WEIGHT="$z" SPECIALIZATION_LOSS_MODE="legacy" sbatch --job-name="m28_${short_task}_${seed}_${tag}" "$JOB_SCRIPT")
      job_id="$(awk '{print $4}' <<< "$submit_output")"
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$job_id" "$config_id" "$task" "$seed" "$sem" "$shared" "$orth" "$var" "$z" "$semonly" >> "$MANIFEST"
      echo "$submit_output $config_id"
    done
  done
done
echo "Manifest: $MANIFEST"
