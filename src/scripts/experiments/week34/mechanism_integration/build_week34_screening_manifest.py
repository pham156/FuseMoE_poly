#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from layout import MANIFEST_ROOT, REPORT_ROOT, ensure_week34_dirs, train_output_dir


TASKS = [
    ("pheno-all-cxr-notes-ecg", "pheno", "macro_f1", 25),
    ("ihm-48-cxr-notes-ecg", "ihm", "f1", 2),
]
SEEDS = [32, 42, 52]


FIELDNAMES = [
    "status",
    "skip_reason",
    "week34_stage",
    "config_id",
    "family_name",
    "family_variant",
    "paper_source",
    "task",
    "task_short",
    "primary_metric",
    "num_labels",
    "architecture",
    "seed",
    "gating_function",
    "poly_power",
    "router_type",
    "router_init",
    "router_init_std",
    "expert_type",
    "lora_rank",
    "freeze_expert_base",
    "disable_run_folder_save",
    "output_dir",
    "implementation_status",
    "planned_module_changes",
    "use_unimodal_kd",
    "unimodal_kd_modalities",
    "unimodal_kd_weight",
    "unimodal_kd_weights",
    "unimodal_teacher_dir",
    "unimodal_kd_temperature",
    "use_full_partial_consistency",
    "consistency_weight",
    "consistency_loss_type",
    "consistency_mask_mode",
    "consistency_missingness_level",
    "use_labelwise_fusion",
    "labelwise_fusion_source",
    "labelwise_hidden_dim",
    "labelwise_rank_loss_weight",
]


def base_row(task: str, task_short: str, primary_metric: str, num_labels: int, architecture: str, seed: int) -> dict[str, str]:
    return {
        "status": "ready",
        "skip_reason": "",
        "week34_stage": "screening",
        "task": task,
        "task_short": task_short,
        "primary_metric": primary_metric,
        "num_labels": str(num_labels),
        "architecture": architecture,
        "seed": str(seed),
        "gating_function": "poly",
        "poly_power": "4",
        "router_type": "permod",
        "router_init": "normal",
        "router_init_std": "0.02",
        "expert_type": "mlp",
        "lora_rank": "8",
        "freeze_expert_base": "False",
        "disable_run_folder_save": "True",
        "implementation_status": "scaffold_only",
        "planned_module_changes": "",
        "use_unimodal_kd": "False",
        "unimodal_kd_modalities": "ts,text,cxr,ecg",
        "unimodal_kd_weight": "1.0",
        "unimodal_kd_weights": "",
        "unimodal_teacher_dir": "",
        "unimodal_kd_temperature": "1.0",
        "use_full_partial_consistency": "False",
        "consistency_weight": "0.0",
        "consistency_loss_type": "mse",
        "consistency_mask_mode": "single_random",
        "consistency_missingness_level": "0.0",
        "use_labelwise_fusion": "False",
        "labelwise_fusion_source": "modalities",
        "labelwise_hidden_dim": "128",
        "labelwise_rank_loss_weight": "0.0",
    }


def family_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    families = [
        (
            "W34_A_unimodal_kd",
            "A1_equal_weight_kd",
            "MIND",
            {
                "status": "planned",
                "skip_reason": "missing_unimodal_teacher_targets",
                "planned_module_changes": "loss,prediction_head,training_schedule",
                "use_unimodal_kd": "True",
                "unimodal_kd_weight": "1.0",
            },
        ),
        (
            "W34_A_unimodal_kd",
            "A2_weighted_kd",
            "MIND",
            {
                "status": "planned",
                "skip_reason": "missing_unimodal_teacher_targets",
                "planned_module_changes": "loss,prediction_head,training_schedule",
                "use_unimodal_kd": "True",
                "unimodal_kd_weight": "1.0",
                "unimodal_kd_weights": "ts=1.0,text=1.0,cxr=1.0,ecg=1.0",
            },
        ),
        (
            "W34_B_consistency",
            "B1_single_mask_consistency",
            "DrFuse+MissingModalityJBI",
            {
                "planned_module_changes": "loss,missing_modality_module",
                "use_full_partial_consistency": "True",
                "consistency_weight": "0.1",
                "consistency_loss_type": "mse",
                "consistency_mask_mode": "single_random",
            },
        ),
        (
            "W34_B_consistency",
            "B2_progressive_consistency",
            "DrFuse+MissingModalityJBI",
            {
                "planned_module_changes": "loss,missing_modality_module",
                "use_full_partial_consistency": "True",
                "consistency_weight": "0.1",
                "consistency_loss_type": "mse",
                "consistency_mask_mode": "progressive",
                "consistency_missingness_level": "0.25",
            },
        ),
        (
            "W34_C_labelwise_fusion",
            "C1_labelwise_modalities",
            "DrFuse+MedPatch",
            {
                "planned_module_changes": "fusion,prediction_head",
                "use_labelwise_fusion": "True",
                "labelwise_fusion_source": "modalities",
                "labelwise_hidden_dim": "128",
            },
        ),
        (
            "W34_C_labelwise_fusion",
            "C2_labelwise_branches",
            "DrFuse+MedPatch",
            {
                "planned_module_changes": "fusion,prediction_head",
                "use_labelwise_fusion": "True",
                "labelwise_fusion_source": "branches",
                "labelwise_hidden_dim": "128",
            },
        ),
    ]

    for task, task_short, primary_metric, num_labels in TASKS:
        for architecture in ("moe_ref", "recon_ref"):
            for seed in SEEDS:
                for family_name, variant_name, paper_source, flags in families:
                    row = base_row(task, task_short, primary_metric, num_labels, architecture, seed)
                    row.update(flags)
                    row["family_name"] = family_name
                    row["family_variant"] = variant_name
                    row["paper_source"] = paper_source
                    row["config_id"] = f"w34_{family_name.split('_')[1].lower()}_{variant_name}_{architecture}_{task_short}_s{seed}"
                    row["output_dir"] = str(train_output_dir(family_name, variant_name, architecture, task_short, seed) / row["config_id"])
                    rows.append(row)
    return rows


def write_notes(rows: list[dict[str, str]]) -> None:
    family_counts: dict[str, int] = {}
    for row in rows:
        family_counts[row["family_name"]] = family_counts.get(row["family_name"], 0) + 1
    notes_path = REPORT_ROOT / "week34_screening_plan.md"
    with notes_path.open("w") as handle:
        handle.write("# Week34 screening plan\n\n")
        handle.write("This manifest is partially runnable. `W34_B_consistency` and `W34_C_labelwise_fusion` are marked `ready`. `W34_A_unimodal_kd` remains `planned` until unimodal teacher-target files are exported.\n\n")
        handle.write("## Family counts\n\n")
        for family_name, count in sorted(family_counts.items()):
            handle.write(f"- `{family_name}`: {count} rows\n")
        handle.write("\n## Tasks\n\n- `pheno-all-cxr-notes-ecg`\n- `ihm-48-cxr-notes-ecg`\n")
        handle.write("\n## Architectures\n\n- `moe_ref`\n- `recon_ref`\n")
        handle.write("\n## Seeds\n\n- `32`\n- `42`\n- `52`\n")


def main() -> None:
    ensure_week34_dirs()
    rows = family_rows()
    manifest_path = MANIFEST_ROOT / "week34_screening_manifest.tsv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    write_notes(rows)
    print(f"Wrote {len(rows)} rows to {manifest_path}")


if __name__ == "__main__":
    main()
