#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/job_lingshu_pseudotoken_week26.sbatch"

SEEDS="1 20 30"
RATIOS="0.05 0.1 0.4 0.8"
EPOCHS=1

for seed in $SEEDS; do
  echo "Submitting Week26 Lingshu pseudo-token diagnostic seed=$seed ratios=$RATIOS"
  sbatch "$JOB_SCRIPT" \
    --seed "$seed" \
    --ratios "$RATIOS" \
    --epochs "$EPOCHS"
done
