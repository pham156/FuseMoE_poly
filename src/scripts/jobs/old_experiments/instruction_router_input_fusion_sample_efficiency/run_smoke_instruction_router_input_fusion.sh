#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "$SCRIPT_DIR/job_instruction_router_input_fusion_sample_efficiency.sbatch" \
  --task ihm-48-cxr-notes-ecg \
  --num_labels 2 \
  --primary_metric f1 \
  --seed 30 \
  --router joint \
  --experts 4 \
  --top_k 2 \
  --disjoint_top_k 2 \
  --config_name instr_input_smoke_joint4_k2 \
  --ratios "0.025" \
  --epochs 1
