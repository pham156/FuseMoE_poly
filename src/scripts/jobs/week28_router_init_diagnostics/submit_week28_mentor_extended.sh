#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/job_week28_mentor_config.sbatch"
OUT_DIR="/home/pham156/MoE/FuseMoE_poly/out/Week_28/mentor_specialization"
MANIFEST="${OUT_DIR}/week28_mentor_extended_manifest.tsv"
mkdir -p "$OUT_DIR"

printf "job_id\tconfig_id\ttask\tseed\tsemantic_scale\tshared_weight\torth\tvar\tz\tsemantic_only\tset\n" > "$MANIFEST"

submit_one() {
  local set_name="$1"
  local task="$2"
  local labels="$3"
  local metric="$4"
  local seed="$5"
  local tag="$6"
  local sem="$7"
  local shared="$8"
  local orth="$9"
  local var="${10}"
  local z="${11}"
  local semonly="${12}"

  local short_task="${task%%-*}"
  [[ "$task" == pheno-* ]] && short_task="pheno"
  local enable_shared="False"
  [[ "$shared" != "0.0" ]] && enable_shared="True"
  local use_sem="False"
  [[ "$sem" != "0.0" || "$semonly" == "True" ]] && use_sem="True"
  local config_id="w28_${set_name}_${short_task}_s${seed}_${tag}"

  local submit_output
  submit_output=$(
    CONFIG_ID="$config_id" \
    TASK="$task" \
    NUM_LABELS="$labels" \
    PRIMARY_METRIC="$metric" \
    SEED="$seed" \
    ROUTER="permod" \
    SHARED_EXPERTS="0" \
    ARCH_VARIANT="random_low_noise_decay" \
    MODELTYPE="TS_CXR_Text" \
    NUM_MODALITIES="3" \
    TOTAL_EPOCHS="16" \
    NUM_EXPERTS="4" \
    TOP_K="2" \
    BALANCE_COEF="0.01" \
    ROUTER_PRINT_MODE="concise" \
    LOG_EXPERT_OUTPUT_DIAGNOSTICS="True" \
    USE_SEMANTIC_LOGIT_BIAS="$use_sem" \
    SEMANTIC_BIAS_SCALE="$sem" \
    SEMANTIC_PROJECT_DIM="128" \
    SEMANTIC_ONLY_ROUTER="$semonly" \
    SEMANTIC_PROFILE_EMBEDDING_SOURCE="random" \
    ENABLE_SHARED_EXPERT="$enable_shared" \
    SHARED_EXPERT_WEIGHT="$shared" \
    ORTHOGONAL_LOSS_WEIGHT="$orth" \
    ROUTER_VARIANCE_LOSS_WEIGHT="$var" \
    Z_LOSS_WEIGHT="$z" \
    SPECIALIZATION_LOSS_MODE="legacy" \
    sbatch --job-name="x28_${short_task}_${seed}_${tag}" "$JOB_SCRIPT"
  )
  local job_id
  job_id="$(awk '{print $4}' <<< "$submit_output")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$job_id" "$config_id" "$task" "$seed" "$sem" "$shared" "$orth" "$var" "$z" "$semonly" "$set_name" >> "$MANIFEST"
  echo "$submit_output $config_id"
}

SEEDS=("32" "42")

# Add pheno coverage for the same mentor configs used on IHM/LOS.
PHENO_CONFIGS=(
  "base 0.0 0.0 0.0 0.0 0.0 False"
  "sem01 0.1 0.0 0.0 0.0 0.0 False"
  "sem03 0.3 0.0 0.0 0.0 0.0 False"
  "sem01sh 0.1 1.0 0.0 0.0 0.0 False"
  "sem01reg 0.1 1.0 0.001 0.001 0.001 False"
  "semonly 1.0 0.0 0.0 0.0 0.0 True"
)
for seed in "${SEEDS[@]}"; do
  for cfg in "${PHENO_CONFIGS[@]}"; do
    read -r tag sem shared orth var z semonly <<< "$cfg"
    submit_one "mentorx" "pheno-all-cxr-notes-ecg" "25" "macro_f1" "$seed" "$tag" "$sem" "$shared" "$orth" "$var" "$z" "$semonly"
  done
done

# Extra IHM/LOS sensitivity around semantic scale, shared path, and regularizer strength.
TASK_SPECS=(
  "ihm-48-cxr-notes-ecg 2 f1"
  "los-48-cxr-notes-ecg 2 f1"
)
EXTRA_CONFIGS=(
  "sem005 0.05 0.0 0.0 0.0 0.0 False"
  "sem03sh 0.3 1.0 0.0 0.0 0.0 False"
  "sem01z003 0.1 1.0 0.0 0.0 0.003 False"
  "sem01orth003 0.1 1.0 0.003 0.0 0.0 False"
  "sem01var003 0.1 1.0 0.0 0.003 0.0 False"
  "sem03reg003 0.3 1.0 0.003 0.003 0.003 False"
)
for task_spec in "${TASK_SPECS[@]}"; do
  read -r task labels metric <<< "$task_spec"
  for seed in "${SEEDS[@]}"; do
    for cfg in "${EXTRA_CONFIGS[@]}"; do
      read -r tag sem shared orth var z semonly <<< "$cfg"
      submit_one "mentorx" "$task" "$labels" "$metric" "$seed" "$tag" "$sem" "$shared" "$orth" "$var" "$z" "$semonly"
    done
  done
done

echo "Manifest: $MANIFEST"
