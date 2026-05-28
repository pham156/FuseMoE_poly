#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/job_softmax_collapse_week26.sbatch"

# Default submits one job that loops over balance_loss_coef = 0.01, 0.1, 1.0.
# Override with env vars if needed, for example:
#   BALANCE_COEFS="0.01" RATIO=0.4 SEED=30 bash run_softmax_collapse_week26.sh

chmod +x "$JOB_SCRIPT"
sbatch "$JOB_SCRIPT"
