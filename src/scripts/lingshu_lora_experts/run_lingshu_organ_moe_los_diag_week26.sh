#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/job_lingshu_organ_moe_los_diag_week26.sbatch"

SEEDS=(1 20 30)

for seed in "${SEEDS[@]}"; do
  echo "Submitting organ-LoRA-MoE LOS diagnostic seed=$seed ratios=0.025,0.05,0.1,0.4,1.0"
  sbatch "$JOB_SCRIPT" --seed "$seed"
done
