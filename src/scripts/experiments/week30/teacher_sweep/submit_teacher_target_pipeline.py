#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


REPO = Path("/home/pham156/MoE/FuseMoE_poly")
SCRIPT_DIR = REPO / "src" / "scripts" / "experiments" / "week30" / "teacher_sweep"
EXPORT_SBATCH = SCRIPT_DIR / "job_teacher_target_export.sbatch"
FINALIZE_SBATCH = SCRIPT_DIR / "job_finalize_teacher.sbatch"
JOBMAP = REPO / "out" / "Week_30" / "teacher_sweep" / "manifests" / "teacher_target_pipeline_jobs.tsv"
DEFAULT_TEACHERS = ("qwen25", "biomistral", "me_llama", "lingshu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teachers", default=",".join(DEFAULT_TEACHERS))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def submit(cmd: list[str], dry_run: bool) -> str:
    if dry_run:
        print(" ".join(cmd))
        return "DRY_RUN"
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip().split()[-1]


def main() -> None:
    args = parse_args()
    teachers = [value.strip() for value in args.teachers.split(",") if value.strip()]
    rows = []
    for teacher in teachers:
        export_job = submit(
            [
                "sbatch",
                f"--job-name=w30target_{teacher}",
                f"--export=ALL,TEACHER_MODEL={teacher}",
                str(EXPORT_SBATCH),
            ],
            args.dry_run,
        )
        finalize_job = submit(
            [
                "sbatch",
                f"--dependency=afterok:{export_job}",
                f"--job-name=w30launch_{teacher}",
                f"--export=ALL,TEACHER_MODEL={teacher}",
                str(FINALIZE_SBATCH),
            ],
            args.dry_run,
        )
        rows.append({"teacher_model": teacher, "export_job_id": export_job, "finalize_job_id": finalize_job})
        print(f"{teacher}\texport={export_job}\tfinalize={finalize_job}")

    JOBMAP.parent.mkdir(parents=True, exist_ok=True)
    with JOBMAP.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["teacher_model", "export_job_id", "finalize_job_id"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {JOBMAP}")


if __name__ == "__main__":
    main()
