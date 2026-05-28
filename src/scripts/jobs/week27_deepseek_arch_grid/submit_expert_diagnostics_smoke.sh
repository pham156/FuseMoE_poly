#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/job_deepseek_arch_config.sbatch"

CONFIG_ID="diag_smoke_ihm_s32_random"
CONFIG_ID="$CONFIG_ID" \
TASK="ihm-48-cxr-notes-ecg" \
NUM_LABELS="2" \
PRIMARY_METRIC="f1" \
SEED="32" \
ROUTER="permod" \
SHARED_EXPERTS="0" \
ARCH_VARIANT="random_router" \
MODELTYPE="TS_CXR_Text" \
NUM_MODALITIES="3" \
TOTAL_EPOCHS="2" \
NUM_EXPERTS="4" \
TOP_K="2" \
BALANCE_COEF="0.01" \
ROUTER_PRINT_MODE="concise" \
LOG_EXPERT_OUTPUT_DIAGNOSTICS="True" \
sbatch --job-name="diag_smoke" "$JOB_SCRIPT"

