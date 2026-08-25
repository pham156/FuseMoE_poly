#!/usr/bin/env python3
import csv
from pathlib import Path

from layout import MANIFEST_ROOT, RESULTS_HOME_ROOT, ensure_week31_dirs, config_output_dir


TASKS = [
    ("pheno-all-cxr-notes-ecg", "macro_f1", 25),
    ("ihm-48-cxr-notes-ecg", "f1", 2),
    ("los-48-cxr-notes-ecg", "f1", 2),
]
ARCHITECTURES = ["moe_ref", "recon_ref"]
SEEDS = [32, 42, 52]
MODELTYPE = "TS_CXR_Text_ECG"
NUM_MODALITIES = 4
GATING_VARIANTS = [
    ("softmax", "", "softmax"),
    ("laplace", "", "laplace"),
    ("poly", "4", "poly4"),
]

RULE_TARGET = Path("/home/pham156/MoE/FuseMoE_poly/out/Week_29/semantic_init_pheno_first/targets/rule/{split}_rule_targets.csv")
CENTROID_TARGET = Path("/home/pham156/MoE/FuseMoE_poly/out/Week_29/semantic_init_pheno_first/targets/centroid/{split}_centroid_targets.csv")
LINGSHU_TARGET = Path("/home/pham156/MoE/FuseMoE_poly/out/Week_30/teacher_sweep/targets/lingshu/{split}_lingshu_targets.csv")
QWEN_TARGET = Path("/home/pham156/MoE/FuseMoE_poly/out/Week_30/teacher_sweep/targets/qwen25/{split}_qwen25_targets.csv")


ROUTER_VARIANTS = {
    "permod_router": {"router_type": "permod"},
    "interaction_router": {
        "router_type": "permod",
        "use_interaction_router": "True",
    },
    "interaction_missing_router": {
        "router_type": "permod",
        "use_interaction_router": "True",
        "use_modality_mask_condition_router": "True",
        "mask_condition_stats": "True",
    },
    "task_router": {
        "router_type": "permod",
        "use_task_condition_router": "True",
        "task_condition_stats": "True",
    },
    "task_interaction_router": {
        "router_type": "permod",
        "use_task_condition_router": "True",
        "task_condition_stats": "True",
        "use_interaction_router": "True",
    },
    "flame_router": {
        "router_type": "permod",
        "use_modality_mask_condition_router": "True",
        "mask_condition_stats": "True",
        "shared_semantic_memory_mode": "learnable",
    },
    "interaction_only_router": {
        "router_type": "permod",
        "use_interaction_router": "True",
        "interaction_router_only": "True",
    },
}


INIT_VARIANTS = {
    "random": {
        "router_init": "normal",
        "router_init_std": "0.02",
    },
    "xavier": {
        "router_init": "xavier",
    },
    "semantic_warm_start": {
        "expert_init_strategy": "fixed_cohort",
        "expert_init_target_path": str(RULE_TARGET),
        "expert_init_epochs": "6",
        "expert_init_soft_targets": "True",
        "expert_init_freeze_router": "True",
        "expert_init_release_schedule": "linear",
    },
    "interaction_warm_start": {
        "expert_init_strategy": "fixed_cohort",
        "expert_init_target_path": str(CENTROID_TARGET),
        "expert_init_epochs": "6",
        "expert_init_soft_targets": "True",
        "expert_init_freeze_router": "True",
        "expert_init_release_schedule": "linear",
    },
    "teacher_warm_start_lingshu": {
        "expert_init_strategy": "fixed_cohort",
        "expert_init_target_path": str(LINGSHU_TARGET),
        "expert_init_epochs": "6",
        "expert_init_soft_targets": "True",
        "expert_init_freeze_router": "True",
        "expert_init_release_schedule": "linear",
        "teacher_model": "lingshu",
        "teacher_sweep": "True",
    },
    "teacher_warm_start_qwen25": {
        "expert_init_strategy": "fixed_cohort",
        "expert_init_target_path": str(QWEN_TARGET),
        "expert_init_epochs": "6",
        "expert_init_soft_targets": "True",
        "expert_init_freeze_router": "True",
        "expert_init_release_schedule": "linear",
        "teacher_model": "qwen25",
        "teacher_sweep": "True",
    },
}


FIELDNAMES = [
    "config_id",
    "status",
    "skip_reason",
    "task",
    "primary_metric",
    "num_labels",
    "architecture",
    "router_family",
    "init_strategy",
    "seed",
    "modeltype",
    "num_modalities",
    "gating_function",
    "poly_power",
    "router_type",
    "router_init",
    "router_init_std",
    "use_task_condition_router",
    "task_condition_stats",
    "use_modality_mask_condition_router",
    "mask_condition_stats",
    "use_interaction_router",
    "interaction_router_only",
    "interaction_router_scale",
    "shared_semantic_memory_mode",
    "shared_semantic_memory_slots",
    "shared_semantic_memory_heads",
    "expert_init_strategy",
    "expert_init_target_path",
    "expert_init_epochs",
    "expert_init_soft_targets",
    "expert_init_freeze_router",
    "expert_init_release_schedule",
    "teacher_sweep",
    "teacher_model",
    "log_router_diagnostics",
    "log_expert_output_diagnostics",
    "log_interaction_features",
    "week31_variant_id",
    "week31_router_family",
    "output_dir",
]


def _target_ready(template: str) -> bool:
    return all(Path(template.format(split=split)).exists() for split in ("train", "val", "test"))


def build_rows():
    for task, primary_metric, num_labels in TASKS:
        task_short = "pheno" if "pheno" in task else ("los" if "los" in task else "ihm")
        for architecture in ARCHITECTURES:
            for router_family, router_flags in ROUTER_VARIANTS.items():
                for gating_function, poly_power, gating_label in GATING_VARIANTS:
                    for init_strategy, init_flags in INIT_VARIANTS.items():
                        status = "ready"
                        skip_reason = ""
                        target_path = init_flags.get("expert_init_target_path", "")
                        if target_path and not _target_ready(target_path):
                            status = "skip"
                            skip_reason = f"missing targets: {target_path}"
                        for seed in SEEDS:
                            config_id = f"w31_{architecture}_{router_family}_{init_strategy}_{task_short}_{gating_label}_s{seed}"
                            row = {
                                "config_id": config_id,
                                "status": status,
                                "skip_reason": skip_reason,
                                "task": task,
                                "primary_metric": primary_metric,
                                "num_labels": num_labels,
                                "architecture": architecture,
                                "router_family": router_family,
                                "init_strategy": init_strategy,
                                "seed": seed,
                                "modeltype": MODELTYPE,
                                "num_modalities": NUM_MODALITIES,
                                "gating_function": gating_function,
                                "poly_power": poly_power,
                                "router_type": "permod",
                                "router_init": "zero",
                                "router_init_std": "0.02",
                                "use_task_condition_router": "False",
                                "task_condition_stats": "False",
                                "use_modality_mask_condition_router": "False",
                                "mask_condition_stats": "False",
                                "use_interaction_router": "False",
                                "interaction_router_only": "False",
                                "interaction_router_scale": "1.0",
                                "shared_semantic_memory_mode": "none",
                                "shared_semantic_memory_slots": "8",
                                "shared_semantic_memory_heads": "1",
                                "expert_init_strategy": "none",
                                "expert_init_target_path": "",
                                "expert_init_epochs": "0",
                                "expert_init_soft_targets": "False",
                                "expert_init_freeze_router": "False",
                                "expert_init_release_schedule": "hard",
                                "teacher_sweep": "False",
                                "teacher_model": "none",
                                "log_router_diagnostics": "True",
                                "log_expert_output_diagnostics": "True",
                                "log_interaction_features": "True" if router_flags.get("use_interaction_router") == "True" else "False",
                                "week31_variant_id": f"{router_family}__{init_strategy}__{gating_label}",
                                "week31_router_family": router_family,
                                "output_dir": str(config_output_dir(architecture, router_family, init_strategy, task_short, seed) / config_id),
                            }
                            row.update(router_flags)
                            row.update(init_flags)
                            yield row


def main() -> None:
    ensure_week31_dirs()
    manifest_path = MANIFEST_ROOT / "week31_train_manifest.tsv"
    summary_path = RESULTS_HOME_ROOT / "reports" / "week31_manifest_summary.md"
    rows = list(build_rows())
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    ready = sum(1 for row in rows if row["status"] == "ready")
    skipped = len(rows) - ready
    summary_lines = [
        "# Week31 Manifest Summary",
        "",
        f"- total_rows: {len(rows)}",
        f"- ready_rows: {ready}",
        f"- skipped_rows: {skipped}",
        f"- manifest: `{manifest_path}`",
        "",
        "## Router Families",
    ]
    for router_family in ROUTER_VARIANTS:
        summary_lines.append(f"- `{router_family}`")
    summary_lines.extend([
        "",
        "## Initialization Variants",
    ])
    for init_strategy in INIT_VARIANTS:
        summary_lines.append(f"- `{init_strategy}`")
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
