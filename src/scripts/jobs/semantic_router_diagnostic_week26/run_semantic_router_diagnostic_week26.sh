#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/job_semantic_router_diagnostic_week26.sbatch"

# Week26 semantic-router diagnostic.
#
# Goal:
#   Test whether Week25 semantic profile routing failed because the semantic
#   bias was too weak/strong, without expanding to the full task/router grid.
#
# Matched baseline config:
#   task: ihm-48-cxr-notes-ecg
#   router: joint
#   experts: 4
#   top_k: 2
#   gating: softmax

SEEDS="1 20 30"
RATIOS="0.05 0.1 0.4 0.8"
EPOCHS=8

# Main sweep: additive semantic bias at multiple scales.
SCALE_FUSION_CONFIGS=(
  "0.1 add sem_diag_add_s0p1"
  "0.3 add sem_diag_add_s0p3"
  "1.0 add sem_diag_add_s1p0"
  "3.0 add sem_diag_add_s3p0"
  "1.0 replace sem_diag_replace_s1p0"
)

for config in "${SCALE_FUSION_CONFIGS[@]}"; do
  read -r scale fusion config_name <<< "$config"

  echo "Submitting Week26 semantic diagnostic scale=$scale fusion=$fusion seeds=$SEEDS ratios=$RATIOS"
  sbatch "$JOB_SCRIPT" \
    --semantic_profile_scale "$scale" \
    --semantic_profile_fusion "$fusion" \
    --config_name "$config_name" \
    --seeds "$SEEDS" \
    --ratios "$RATIOS" \
    --epochs "$EPOCHS"
done
