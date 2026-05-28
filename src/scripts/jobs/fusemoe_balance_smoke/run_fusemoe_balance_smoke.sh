#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/job_fusemoe_balance_smoke.sbatch"

echo "Submitting one FuseMoE balance 8epoch grid job for ratios=0.2,0.4,0.6,0.8,1.0"
sbatch "$JOB_SCRIPT"
