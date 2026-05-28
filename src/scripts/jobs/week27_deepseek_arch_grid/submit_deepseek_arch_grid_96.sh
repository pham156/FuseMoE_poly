#!/bin/bash
set -eo pipefail

ROOT="/home/pham156/MoE/FuseMoE_poly"
JOB_SCRIPT="${ROOT}/src/scripts/week27_deepseek_arch_grid/job_deepseek_arch_config.sbatch"
OUT_DIR="${ROOT}/out/Week_27/deepseek_arch_grid"
MANIFEST="${OUT_DIR}/deepseek_arch_grid_manifest.tsv"

mkdir -p "$OUT_DIR"
printf "job_id\tconfig_id\ttask\tseed\trouter\tshared\tarch_variant\n" > "$MANIFEST"

TASK_SPECS=(
  "ihm-48-cxr-notes-ecg 2 f1 ihm"
  "los-48-cxr-notes-ecg 2 f1 los"
)
SEEDS=(32 42)
ROUTERS=(joint permod)
SHARED=(0 1)
ARCH_VARIANTS=(
  fusemoe_original
  zero_no_noise
  random_router
  deepseek_gate
  deepseek_aux
  deepseek_hidden_aux
  xmoe_hidden_aux
)

submit_one() {
  local task="$1"
  local labels="$2"
  local metric="$3"
  local task_tag="$4"
  local seed="$5"
  local router="$6"
  local shared="$7"
  local variant="$8"

  local config_id="dsarch_${task_tag}_s${seed}_${router}_sh${shared}_${variant}"
  local job_name="da_${task_tag}_s${seed}_${router:0:1}${shared}_${variant:0:8}"

  local job_id
  job_id=$(sbatch --parsable \
    --job-name "$job_name" \
    --export=ALL,CONFIG_ID="$config_id",TASK="$task",NUM_LABELS="$labels",PRIMARY_METRIC="$metric",SEED="$seed",ROUTER="$router",SHARED_EXPERTS="$shared",ARCH_VARIANT="$variant" \
    "$JOB_SCRIPT")

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$job_id" "$config_id" "$task" "$seed" "$router" "$shared" "$variant" >> "$MANIFEST"
  echo "submitted ${job_id} ${config_id}"
}

count=0
for spec in "${TASK_SPECS[@]}"; do
  read -r task labels metric task_tag <<< "$spec"
  for seed in "${SEEDS[@]}"; do
    for router in "${ROUTERS[@]}"; do
      for shared in "${SHARED[@]}"; do
        for variant in "${ARCH_VARIANTS[@]}"; do
          submit_one "$task" "$labels" "$metric" "$task_tag" "$seed" "$router" "$shared" "$variant"
          count=$((count + 1))
        done
      done
    done
  done
done

echo "submitted_total=${count}"
echo "manifest=${MANIFEST}"
