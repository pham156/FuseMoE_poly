#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$SCRIPT_DIR/job_lingshu_pseudotoken_smoke.sbatch"
