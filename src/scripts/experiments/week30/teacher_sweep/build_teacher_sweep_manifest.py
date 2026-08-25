#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.append(str(SCRIPT_DIR.parent))

from scripts.experiments.week30.teacher_sweep.teacher_registry import MODEL_ORDER, get_teacher_spec


HOME_ROOT = Path("/home/pham156/MoE/FuseMoE_poly")
SCRATCH_ROOT = Path("/scratch/gilbreth/pham156/MoE/FuseMoE_poly")
OUT_HOME = HOME_ROOT / "out" / "Week_30" / "teacher_sweep"
OUT_SCRATCH = SCRATCH_ROOT / "out" / "Week_30" / "teacher_sweep"

TASK = "pheno-all-cxr-notes-ecg"
PRIMARY_METRIC = "macro_f1"
NUM_LABELS = 25
MODELTYPE = "TS_CXR_Text_ECG"
NUM_MODALITIES = 4
SEEDS = [32, 42, 52]
ARCHS = ["moe_ref", "recon_ref"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-file", default=str(OUT_HOME / "manifests" / "teacher_target_status.tsv"))
    parser.add_argument("--manifest", default=str(OUT_HOME / "manifests" / "teacher_sweep_pheno_manifest.tsv"))
    return parser.parse_args()


def _load_status(path: Path) -> dict[str, dict]:
    rows = {}
    if not path.exists():
        return rows
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows[row["teacher_model"]] = row
    return rows


def _target_path(source_key: str, seed: int) -> str:
    target_root = OUT_HOME / "targets"
    if source_key == "random":
        return str(target_root / "random" / f"seed_{seed}" / "{split}_random_targets.csv")
    return str(target_root / source_key / f"{{split}}_{source_key}_targets.csv")


def _output_dir(arch: str, seed: int, teacher_model: str) -> Path:
    return OUT_SCRATCH / f"{arch}_{teacher_model}" / f"seed_{seed}"


def main() -> None:
    args = parse_args()
    status = _load_status(Path(args.status_file))
    fieldnames = [
        "config_id",
        "architecture",
        "task",
        "seed",
        "primary_metric",
        "num_labels",
        "modeltype",
        "num_modalities",
        "teacher_model",
        "teacher_source",
        "teacher_family",
        "teacher_available",
        "teacher_skip_reason",
        "expert_init_strategy",
        "expert_init_target_path",
        "expert_init_epochs",
        "expert_init_soft_targets",
        "expert_init_freeze_router",
        "expert_init_release_schedule",
        "should_submit",
        "output_dir",
    ]
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for arch in ARCHS:
            for seed in SEEDS:
                for teacher_model in MODEL_ORDER:
                    spec = get_teacher_spec(teacher_model)
                    st = status.get(teacher_model, {})
                    status_value = st.get("status", "")
                    available = teacher_model == "none" or status_value in {"ready", "available_not_exported"}
                    should_submit = teacher_model == "none" or status_value == "ready"
                    config_id = f"w30t_{arch}_pheno_s{seed}_{spec.source}"
                    row = {
                        "config_id": config_id,
                        "architecture": arch,
                        "task": TASK,
                        "seed": seed,
                        "primary_metric": PRIMARY_METRIC,
                        "num_labels": NUM_LABELS,
                        "modeltype": MODELTYPE,
                        "num_modalities": NUM_MODALITIES,
                        "teacher_model": teacher_model,
                        "teacher_source": spec.source,
                        "teacher_family": spec.teacher_family,
                        "teacher_available": str(bool(available)),
                        "teacher_skip_reason": st.get("reason", "") if not available else "",
                        "expert_init_strategy": "none" if teacher_model == "none" else "fixed_cohort",
                        "expert_init_target_path": "" if teacher_model == "none" else _target_path(teacher_model, seed),
                        "expert_init_epochs": 0 if teacher_model == "none" else 6,
                        "expert_init_soft_targets": str(teacher_model != "none"),
                        "expert_init_freeze_router": str(teacher_model != "none"),
                        "expert_init_release_schedule": "" if teacher_model == "none" else "linear",
                        "should_submit": str(bool(should_submit)),
                        "output_dir": str(_output_dir(arch, seed, teacher_model)),
                    }
                    writer.writerow(row)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
