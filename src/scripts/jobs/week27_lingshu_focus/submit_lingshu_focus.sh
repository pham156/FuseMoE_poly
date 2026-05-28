#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/job_lingshu_focus.sbatch"

MODE="${1:-smoke}"

case "$MODE" in
  smoke)
    echo "Submitting one Lingshu organ-LoRA-MoE smoke job."
    sbatch \
      --export=ALL,OUT_TAG=smoke_linear,ARCH=organ_lora_moe,ROUTER=linear,TASK=ihm-48-cxr-notes-ecg,NUM_LABELS=2,PRIMARY_METRIC=f1,SEED=32,RATIO=0.025,EPOCHS=1,MODELTYPE=TS_CXR_Text_ECG,NUM_MODALITIES=4 \
      "$JOB_SCRIPT"
    ;;
  pseudo-smoke)
    echo "Submitting one Lingshu pseudo-token smoke job."
    sbatch \
      --export=ALL,OUT_TAG=pseudo_smoke,ARCH=pseudotoken,ROUTER=linear,TASK=ihm-48-cxr-notes-ecg,NUM_LABELS=2,PRIMARY_METRIC=f1,SEED=32,RATIO=0.025,EPOCHS=1,MODELTYPE=TS_CXR_Text_ECG,NUM_MODALITIES=4 \
      "$JOB_SCRIPT"
    ;;
  sanity)
    echo "Submitting one Lingshu organ-LoRA-MoE sanity job. This is the first non-smoke run."
    sbatch \
      --export=ALL,OUT_TAG=sanity_linear,ARCH=organ_lora_moe,ROUTER=linear,TASK=ihm-48-cxr-notes-ecg,NUM_LABELS=2,PRIMARY_METRIC=f1,SEED=32,RATIO=0.1,EPOCHS=3,MODELTYPE=TS_CXR_Text_ECG,NUM_MODALITIES=4 \
      "$JOB_SCRIPT"
    ;;
  full-ihm)
    echo "Submitting one Lingshu organ-LoRA-MoE full IHM job."
    sbatch \
      --export=ALL,OUT_TAG=full_ihm_linear,ARCH=organ_lora_moe,ROUTER=linear,TASK=ihm-48-cxr-notes-ecg,NUM_LABELS=2,PRIMARY_METRIC=f1,SEED=32,RATIO=1.0,EPOCHS=8,MODELTYPE=TS_CXR_Text_ECG,NUM_MODALITIES=4,TRAIN_BS=1,EVAL_BS=4 \
      "$JOB_SCRIPT"
    ;;
  *)
    echo "Usage: $0 {smoke|pseudo-smoke|sanity|full-ihm}" >&2
    exit 1
    ;;
esac
