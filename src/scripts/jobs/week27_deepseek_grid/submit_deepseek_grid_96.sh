#!/bin/bash
set -eo pipefail

ROOT="/home/pham156/MoE/FuseMoE_poly"
JOB_SCRIPT="${ROOT}/src/scripts/week27_deepseek_grid/job_deepseek_grid_config.sbatch"
OUT_DIR="${ROOT}/out/Week_27/deepseek_grid"
MANIFEST="${OUT_DIR}/deepseek_grid_manifest.tsv"

mkdir -p "$OUT_DIR"
printf "job_id\tconfig_id\ttask\tseed\tgate\trouter\tshared\txmoe\tnoise\n" > "$MANIFEST"

TASK_SPECS=(
  "ihm-48-cxr-notes-ecg 2 f1 ihm"
  "los-48-cxr-notes-ecg 2 f1 los"
)
SEEDS=(32 42)
ROUTERS=(joint permod)
SHARED=(0 1)
XMOE_FLAGS=(False True)

submit_one() {
  local task="$1"
  local labels="$2"
  local metric="$3"
  local task_tag="$4"
  local seed="$5"
  local gate="$6"
  local router="$7"
  local shared="$8"
  local xmoe="$9"
  local noise="${10}"

  local arch="base"
  if [[ "$xmoe" == "True" ]]; then
    arch="xmoe"
  fi
  local config_id="ds_${task_tag}_s${seed}_${gate}_${arch}_sh${shared}_${router}_${noise}"
  local job_name="dg_${gate:0:2}_${task_tag}_s${seed}_${router:0:1}${shared}_${arch:0:1}_${noise:0:1}"

  local job_id
  job_id=$(sbatch --parsable \
    --job-name "$job_name" \
    --export=ALL,CONFIG_ID="$config_id",TASK="$task",NUM_LABELS="$labels",PRIMARY_METRIC="$metric",SEED="$seed",GATING_FUNCTION="$gate",ROUTER="$router",SHARED_EXPERTS="$shared",USE_XMOE_ROUTER="$xmoe",NOISE_REGIME="$noise" \
    "$JOB_SCRIPT")

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$job_id" "$config_id" "$task" "$seed" "$gate" "$router" "$shared" "$xmoe" "$noise" >> "$MANIFEST"
  echo "submitted ${job_id} ${config_id}"
}

count=0

# Core doc-aligned sweep: softmax with/without router noise.
for spec in "${TASK_SPECS[@]}"; do
  read -r task labels metric task_tag <<< "$spec"
  for seed in "${SEEDS[@]}"; do
    for router in "${ROUTERS[@]}"; do
      for shared in "${SHARED[@]}"; do
        for xmoe in "${XMOE_FLAGS[@]}"; do
          for noise in none original; do
            submit_one "$task" "$labels" "$metric" "$task_tag" "$seed" softmax "$router" "$shared" "$xmoe" "$noise"
            count=$((count + 1))
          done
        done
      done
    done
  done
done

# Distance-gate control: laplace without noise, to isolate router geometry from noisy exploration.
for spec in "${TASK_SPECS[@]}"; do
  read -r task labels metric task_tag <<< "$spec"
  for seed in "${SEEDS[@]}"; do
    for router in "${ROUTERS[@]}"; do
      for shared in "${SHARED[@]}"; do
        for xmoe in "${XMOE_FLAGS[@]}"; do
          submit_one "$task" "$labels" "$metric" "$task_tag" "$seed" laplace "$router" "$shared" "$xmoe" none
          count=$((count + 1))
        done
      done
    done
  done
done

echo "submitted_total=${count}"
echo "manifest=${MANIFEST}"
