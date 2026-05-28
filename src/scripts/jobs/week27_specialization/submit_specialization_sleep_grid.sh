#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/job_specialization_grid.sbatch"
OUT_DIR="/home/pham156/MoE/FuseMoE_poly/out/Week_27/specialization"
MANIFEST="$OUT_DIR/specialization_sleep_grid_manifest.tsv"
mkdir -p "$OUT_DIR"

EPOCHS="${EPOCHS:-50}"
ROUTER="${ROUTER:-permod}"
GATING_FUNCTION="${GATING_FUNCTION:-softmax}"

TASKS=(
  "ihm-48-cxr-notes-ecg 2 f1"
  "los-48-cxr-notes-ecg 2 f1"
  "pheno-all-cxr-notes-ecg 25 macro_f1"
)
SEEDS=(32 42 52)

echo -e "job_id\ttag\ttask\tseed\tepochs\tspecialization_loss_mode\tspecialization_aux_coef\trouter_variance_coef\toutput_orth_coef\tnoise_start\tnoise_final\tnoise_decay\tdense\tout\terr" > "$MANIFEST"

submit_one() {
  local tag="$1"
  local task="$2"
  local num_labels="$3"
  local primary_metric="$4"
  local seed="$5"
  local bal="$6"
  local var="$7"
  local orth="$8"
  local dense="$9"
  local mode="${10:-paper}"
  local aux="${11:-0.001}"
  local job_name="w27_${tag}_${seed}_${task%%-*}"
  local export_args="ALL,TASK=${task},NUM_LABELS=${num_labels},PRIMARY_METRIC=${primary_metric},SEED=${seed},ROUTER=${ROUTER},GATING_FUNCTION=${GATING_FUNCTION},EPOCHS=${EPOCHS},BALANCE_COEF=${bal},SPECIALIZATION_LOSS_MODE=${mode},SPECIALIZATION_AUX_COEF=${aux},ROUTER_VARIANCE_COEF=${var},OUTPUT_ORTH_COEF=${orth},ROUTER_NOISE_SCALE=0.2,ROUTER_NOISE_FINAL_SCALE=0.02,ROUTER_NOISE_DECAY_EPOCHS=8,DENSE_ROUTING=${dense}"
  local job_id
  job_id=$(sbatch --parsable --job-name="$job_name" --export="$export_args" "$JOB_SCRIPT")
  local out_path="$OUT_DIR/${job_id}_${job_name}.out"
  local err_path="$OUT_DIR/${job_id}_${job_name}.err"
  echo -e "${job_id}\t${tag}\t${task}\t${seed}\t${EPOCHS}\t${mode}\t${aux}\t${var}\t${orth}\t0.2\t0.02\t8\t${dense}\t${out_path}\t${err_path}" | tee -a "$MANIFEST"
}

for task_spec in "${TASKS[@]}"; do
  read -r task num_labels primary_metric <<< "$task_spec"
  for seed in "${SEEDS[@]}"; do
    submit_one "base" "$task" "$num_labels" "$primary_metric" "$seed" 1.0 0.0 0.0 False legacy 0.001
    submit_one "paper001" "$task" "$num_labels" "$primary_metric" "$seed" 1.0 0.001 0.001 False paper 0.001
    submit_one "paper003" "$task" "$num_labels" "$primary_metric" "$seed" 1.0 0.003 0.003 False paper 0.001
    submit_one "dense001" "$task" "$num_labels" "$primary_metric" "$seed" 1.0 0.001 0.001 True paper 0.001
  done
done

echo "Wrote $MANIFEST"
