#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


HOME_ROOT = Path("/home/pham156/MoE/FuseMoE_poly")
MANIFEST = HOME_ROOT / "out" / "Week_30" / "teacher_sweep" / "manifests" / "teacher_sweep_pheno_manifest.tsv"
JOBMAP = HOME_ROOT / "out" / "Week_30" / "teacher_sweep" / "manifests" / "teacher_sweep_pheno_jobs.tsv"
SBATCH = HOME_ROOT / "src" / "scripts" / "experiments" / "week30" / "teacher_sweep" / "job_teacher_sweep.sbatch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--jobmap", default=str(JOBMAP))
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(Path(args.manifest).open(), delimiter="\t"))
    out_rows = []
    for row in rows:
        if str(row.get("should_submit", "")).lower() != "true":
            continue
        if args.teacher_model and row.get("teacher_model") != args.teacher_model:
            continue
        export_items = {
            "CONFIG_ID": row["config_id"],
            "TASK": row["task"],
            "NUM_LABELS": row["num_labels"],
            "PRIMARY_METRIC": row["primary_metric"],
            "SEED": row["seed"],
            "ARCHITECTURE": row["architecture"],
            "MODELTYPE": row["modeltype"],
            "NUM_MODALITIES": row["num_modalities"],
            "OUTPUT_DIR": row["output_dir"],
            "EXPERT_INIT_STRATEGY": row["expert_init_strategy"],
            "EXPERT_INIT_TARGET_PATH": row["expert_init_target_path"],
            "EXPERT_INIT_EPOCHS": row["expert_init_epochs"],
            "EXPERT_INIT_RELEASE_SCHEDULE": row["expert_init_release_schedule"],
            "EXPERT_INIT_SOFT_TARGETS": row["expert_init_soft_targets"],
            "EXPERT_INIT_FREEZE_ROUTER": row["expert_init_freeze_router"],
            "TEACHER_MODEL": row["teacher_model"],
            "TEACHER_FAMILY": row["teacher_family"],
        }
        export_str = ",".join(f"{k}={v}" for k, v in export_items.items())
        cmd = ["sbatch", f"--job-name={row['config_id']}", f"--export={export_str}", str(SBATCH)]
        if args.dry_run:
            job_id = "DRY_RUN"
        else:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split()[-1]
        out_rows.append(
            {
                "config_id": row["config_id"],
                "teacher_model": row["teacher_model"],
                "teacher_family": row["teacher_family"],
                "architecture": row["architecture"],
                "seed": row["seed"],
                "job_id": job_id,
            }
        )
        print(f"{row['config_id']}\t{job_id}")
    jobmap = Path(args.jobmap)
    jobmap.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    if args.append and jobmap.exists():
        with jobmap.open() as handle:
            existing_rows = list(csv.DictReader(handle, delimiter="\t"))
    combined = existing_rows + out_rows
    with jobmap.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config_id", "teacher_model", "teacher_family", "architecture", "seed", "job_id"], delimiter="\t")
        writer.writeheader()
        writer.writerows(combined)
    print(f"Wrote {jobmap}")


if __name__ == "__main__":
    main()
