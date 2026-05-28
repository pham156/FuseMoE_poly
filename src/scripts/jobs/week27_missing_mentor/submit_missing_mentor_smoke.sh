#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/job_missing_mentor_config.sbatch"
OUT_DIR="/home/pham156/MoE/FuseMoE_poly/out/Week_27/missing_mentor"
MANIFEST="${OUT_DIR}/missing_mentor_manifest.tsv"

DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-32}"

mkdir -p "$OUT_DIR"
echo -e "job_id\tconfig_id\tseed\tlearned_missing\trecon\texpert_orth\tout\terr" > "$MANIFEST"

submit_one() {
  local config_id="$1"
  local learned_missing="$2"
  local recon="$3"
  local expert_orth="$4"

  local out_path="${OUT_DIR}/%j_${config_id}.out"
  local err_path="${OUT_DIR}/%j_${config_id}.err"
  local export_args="ALL,CONFIG_ID=${config_id},SEED=${SEED},USE_LEARNED_MISSING=${learned_missing},USE_RECON=${recon},USE_EXPERT_ORTH=${expert_orth},TOTAL_EPOCHS=8,BALANCE_COEF=0.01,ROUTER_ENTROPY_COEF=0.01,RECON_COEF=0.05,EXPERT_ORTH_COEF=0.001"
  local sbatch_cmd=(sbatch --parsable --job-name "w27_${config_id}" --output "$out_path" --error "$err_path" --export "$export_args" "$JOB_SCRIPT")

  local job_id
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN ${config_id}: ${sbatch_cmd[*]}"
    job_id="DRYRUN"
  else
    job_id="$("${sbatch_cmd[@]}")"
  fi

  echo -e "${job_id}\t${config_id}\t${SEED}\t${learned_missing}\t${recon}\t${expert_orth}\t${out_path}\t${err_path}" | tee -a "$MANIFEST"
}

submit_one "base_zero" 0 0 0
submit_one "miss_embed" 1 0 0
submit_one "miss_embed_recon" 1 1 0
submit_one "miss_embed_recon_orth" 1 1 1

echo "Manifest: ${MANIFEST}"

