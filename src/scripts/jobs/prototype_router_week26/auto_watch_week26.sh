#!/bin/bash
set -u

ROOT="/home/pham156/MoE/FuseMoE_poly"
SCRIPT_DIR="$ROOT/src/scripts/prototype_router_week26"
OUT_DIR="$ROOT/out/Week_26"
PROTO_DIR="$OUT_DIR/prototype_router"
WATCH_DIR="$OUT_DIR/auto_watch"
LOG="$WATCH_DIR/auto_watch_$(date +%Y%m%d_%H%M%S).log"
MARKER="$WATCH_DIR/l0_grid_submitted.marker"

mkdir -p "$WATCH_DIR"

{
  echo "===== Week26 auto-watch started $(date) ====="
  echo "log: $LOG"
  echo "This script checks hourly, submits the next predefined layer0 organ-supervision grid once, and refreshes alignment reports."
} | tee -a "$LOG"

for ITER in $(seq 1 12); do
  {
    echo ""
    echo "===== ITER $ITER $(date) ====="
    echo "--- squeue ---"
    squeue -u pham156 || true

    if [[ ! -f "$MARKER" ]]; then
      echo "--- submitting layer0 organ-supervised grid ---"
      JOB_OUTPUT=$(sbatch "$SCRIPT_DIR/job_prototype_router_organ_l0_grid_week26.sbatch" 2>&1)
      echo "$JOB_OUTPUT"
      echo "$JOB_OUTPUT" > "$MARKER"
    else
      echo "--- layer0 grid already submitted ---"
      cat "$MARKER"
    fi

    echo "--- latest prototype logs ---"
    ls -1t "$OUT_DIR"/*proto*.out 2>/dev/null | head -8 || true
    for LOGFILE in $(ls -1t "$OUT_DIR"/*proto*.out 2>/dev/null | head -6); do
      echo ""
      echo "### $(basename "$LOGFILE")"
      rg -n "FINAL TEST RESULTS|auc:|auprc:|f1:|Best validation|Finished" "$LOGFILE" || true
    done

    echo "--- refreshing alignment reports ---"
    python "$SCRIPT_DIR/analyze_prototype_alignment.py" \
      --input_glob "$PROTO_DIR/*router_diagnostics.csv" \
      --output_dir "$PROTO_DIR/alignment_auto" || true

    echo "--- available diagnostics ---"
    find "$PROTO_DIR" -maxdepth 1 -type f -name "*router_diagnostics.csv" -printf "%f\n" | sort || true
  } >> "$LOG" 2>&1

  sleep 3600
done

echo "===== Week26 auto-watch finished $(date) =====" >> "$LOG"
