#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/job_deepseek_arch_config.sbatch"
MANIFEST="/home/pham156/MoE/FuseMoE_poly/out/Week_27/deepseek_arch_grid/expert_diagnostics_manifest.tsv"
mkdir -p "$(dirname "$MANIFEST")"
printf "job_id\tconfig_id\ttask\tseed\trouter\tshared_experts\tarch_variant\tmodeltype\tepochs\n" > "$MANIFEST"

TASKS=("ihm-48-cxr-notes-ecg" "los-48-cxr-notes-ecg")
METRICS=("f1" "f1")
SEEDS=("32" "42")
VARIANTS=("fusemoe_original" "zero_no_noise" "random_router" "random_low_noise_decay")

for task_idx in "${!TASKS[@]}"; do
  task="${TASKS[$task_idx]}"
  metric="${METRICS[$task_idx]}"
  for seed in "${SEEDS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      short_task="${task%%-*}"
      case "$variant" in
        fusemoe_original) short_variant="orig" ;;
        zero_no_noise) short_variant="znon" ;;
        random_router) short_variant="rand" ;;
        random_low_noise_decay) short_variant="rndec" ;;
        *) short_variant="$variant" ;;
      esac
      config_id="diag_${short_task}_s${seed}_${short_variant}"
      submit_output=$(
        CONFIG_ID="$config_id" \
        TASK="$task" \
        NUM_LABELS="2" \
        PRIMARY_METRIC="$metric" \
        SEED="$seed" \
        ROUTER="permod" \
        SHARED_EXPERTS="0" \
        ARCH_VARIANT="$variant" \
        MODELTYPE="TS_CXR_Text" \
        NUM_MODALITIES="3" \
        TOTAL_EPOCHS="16" \
        NUM_EXPERTS="4" \
        TOP_K="2" \
        BALANCE_COEF="0.01" \
        ROUTER_PRINT_MODE="concise" \
        LOG_EXPERT_OUTPUT_DIAGNOSTICS="True" \
        sbatch --job-name="dg_${short_task}_${seed}_${short_variant}" "$JOB_SCRIPT"
      )
      job_id="$(awk '{print $4}' <<< "$submit_output")"
      printf "%s\t%s\t%s\t%s\tpermod\t0\t%s\tTS_CXR_Text\t16\n" "$job_id" "$config_id" "$task" "$seed" "$variant" >> "$MANIFEST"
      echo "$submit_output $config_id"
    done
  done
done

echo "Manifest: $MANIFEST"

