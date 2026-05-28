#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/job_specialization_grid.sbatch"
OUT_DIR="/home/pham156/MoE/FuseMoE_poly/out/Week_27/specialization"
MANIFEST="$OUT_DIR/specialization_grid_manifest.tsv"
mkdir -p "$OUT_DIR"

SEED="${SEED:-32}"
TASK="${TASK:-ihm-48-cxr-notes-ecg}"
ROUTER="${ROUTER:-permod}"
GATING_FUNCTION="${GATING_FUNCTION:-softmax}"

echo -e "job_id\ttag\tepochs\tspecialization_loss_mode\tspecialization_aux_coef\trouter_variance_coef\toutput_orth_coef\tnoise_start\tnoise_final\tnoise_decay\tdense\tout\terr" > "$MANIFEST"

submit_one() {
  local tag="$1"
  local epochs="$2"
  local bal="$3"
  local var="$4"
  local orth="$5"
  local n0="$6"
  local n1="$7"
  local decay="$8"
  local dense="$9"
  local mode="${10:-paper}"
  local aux="${11:-0.001}"
  local job_name="w27_${tag}"
  local export_args="ALL,TASK=${TASK},SEED=${SEED},ROUTER=${ROUTER},GATING_FUNCTION=${GATING_FUNCTION},EPOCHS=${epochs},BALANCE_COEF=${bal},SPECIALIZATION_LOSS_MODE=${mode},SPECIALIZATION_AUX_COEF=${aux},ROUTER_VARIANCE_COEF=${var},OUTPUT_ORTH_COEF=${orth},ROUTER_NOISE_SCALE=${n0},ROUTER_NOISE_FINAL_SCALE=${n1},ROUTER_NOISE_DECAY_EPOCHS=${decay},DENSE_ROUTING=${dense}"
  local job_id
  job_id=$(sbatch --parsable --job-name="$job_name" --export="$export_args" "$JOB_SCRIPT")
  local out_path="$OUT_DIR/${job_id}_${job_name}.out"
  local err_path="$OUT_DIR/${job_id}_${job_name}.err"
  echo -e "${job_id}\t${tag}\t${epochs}\t${mode}\t${aux}\t${var}\t${orth}\t${n0}\t${n1}\t${decay}\t${dense}\t${out_path}\t${err_path}" | tee -a "$MANIFEST"
}

# Cheap evidence grid: one task, one seed. Use 16 epochs because one log already
# captures the 4/8/16 epoch trajectory, and old logs show this fits in standby.
submit_one "base_ep16" 16 1.0 0.0 0.0 0.2 0.02 8 False legacy 0.001
submit_one "paper001_ep16" 16 1.0 0.001 0.001 0.2 0.02 8 False paper 0.001
submit_one "paper003_ep16" 16 1.0 0.003 0.003 0.2 0.02 8 False paper 0.001
submit_one "dense001_ep16" 16 1.0 0.001 0.001 0.2 0.02 8 True paper 0.001

echo "Wrote $MANIFEST"
