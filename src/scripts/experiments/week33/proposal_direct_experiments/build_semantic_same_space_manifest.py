#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from layout import MANIFEST_ROOT, WEEK33_SCRATCH, ensure_week33_dirs


TASKS = [
    ("pheno-all-cxr-notes-ecg", "pheno", "macro_f1", 25),
    ("ihm-48-cxr-notes-ecg", "ihm", "f1", 2),
    ("los-48-cxr-notes-ecg", "los", "f1", 2),
]
SEEDS = [32, 42, 52]

FIELDNAMES = [
    "status",
    "skip_reason",
    "config_id",
    "experiment_group",
    "variant_name",
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
    "staged_shared_lora",
    "staged_shared_lora_warmup_epochs",
    "expert_orth_coef",
    "use_semantic_expert_profiles",
    "use_semantic_logit_bias",
    "semantic_only_router",
    "semantic_bias_scale",
    "semantic_profile_set",
    "semantic_profile_source",
    "semantic_profile_fusion",
    "semantic_profile_note_pooling",
    "semantic_profile_modalities",
    "semantic_profile_layers",
    "semantic_profile_embedding_source",
    "semantic_profile_shuffle_assignments",
    "use_router_organ_supervision",
    "router_organ_supervision_coef",
    "router_organ_supervision_class_balanced",
    "use_missing_modality_recon",
    "missing_modality_recon_coef",
    "missing_modality_recon_targets",
    "missing_modality_recon_hidden",
    "use_missing_modality_proxies",
    "use_cross_modal_missing_proxies",
    "cross_modal_proxy_hidden",
    "cross_modal_proxy_dropout",
    "missing_modality_proxy_init_std",
    "train_cxr_modality_dropout",
    "train_text_modality_dropout",
    "train_ecg_modality_dropout",
    "expert_init_strategy",
    "expert_init_target_path",
    "expert_init_epochs",
    "expert_init_soft_targets",
    "expert_init_freeze_router",
    "expert_init_release_schedule",
    "disable_run_folder_save",
    "output_dir",
]


def output_dir(variant_name: str, task_short: str, seed: int) -> Path:
    return (
        WEEK33_SCRATCH
        / "semantic_same_space_permod"
        / variant_name
        / "moe_ref"
        / task_short
        / f"seed_{seed}"
    )


def base_row(task: str, task_short: str, primary_metric: str, num_labels: int, seed: int) -> dict[str, str]:
    return {
        "status": "ready",
        "skip_reason": "",
        "task": task,
        "task_short": task_short,
        "primary_metric": primary_metric,
        "num_labels": str(num_labels),
        "architecture": "moe_ref",
        "seed": str(seed),
        "gating_function": "poly",
        "poly_power": "4",
        "router_type": "permod",
        "router_init": "normal",
        "router_init_std": "0.02",
        "expert_type": "mlp",
        "lora_rank": "8",
        "freeze_expert_base": "True",
        "staged_shared_lora": "False",
        "staged_shared_lora_warmup_epochs": "8",
        "expert_orth_coef": "0.0",
        "use_semantic_expert_profiles": "False",
        "use_semantic_logit_bias": "False",
        "semantic_only_router": "False",
        "semantic_bias_scale": "",
        "semantic_profile_set": "icu_organ_system",
        "semantic_profile_source": "note",
        "semantic_profile_fusion": "add",
        "semantic_profile_note_pooling": "max",
        "semantic_profile_modalities": "txt",
        "semantic_profile_layers": "all",
        "semantic_profile_embedding_source": "biolongformer",
        "semantic_profile_shuffle_assignments": "False",
        "use_router_organ_supervision": "False",
        "router_organ_supervision_coef": "1.0",
        "router_organ_supervision_class_balanced": "False",
        "use_missing_modality_recon": "False",
        "missing_modality_recon_coef": "0.1",
        "missing_modality_recon_targets": "cxr,ecg",
        "missing_modality_recon_hidden": "256",
        "use_missing_modality_proxies": "False",
        "use_cross_modal_missing_proxies": "False",
        "cross_modal_proxy_hidden": "256",
        "cross_modal_proxy_dropout": "0.1",
        "missing_modality_proxy_init_std": "0.02",
        "train_cxr_modality_dropout": "0.0",
        "train_text_modality_dropout": "0.0",
        "train_ecg_modality_dropout": "0.0",
        "expert_init_strategy": "none",
        "expert_init_target_path": "",
        "expert_init_epochs": "0",
        "expert_init_soft_targets": "False",
        "expert_init_freeze_router": "False",
        "expert_init_release_schedule": "hard",
        "disable_run_folder_save": "True",
    }


VARIANTS = [
    ("baseline_poly4", {}),
    ("note_add_txt_scale005", {
        "use_semantic_expert_profiles": "True",
        "use_semantic_logit_bias": "True",
        "semantic_bias_scale": "0.05",
    }),
    ("note_add_txt_scale010", {
        "use_semantic_expert_profiles": "True",
        "use_semantic_logit_bias": "True",
        "semantic_bias_scale": "0.10",
    }),
    ("note_add_txt_scale020", {
        "use_semantic_expert_profiles": "True",
        "use_semantic_logit_bias": "True",
        "semantic_bias_scale": "0.20",
    }),
    ("note_add_txt_scale010_first", {
        "use_semantic_expert_profiles": "True",
        "use_semantic_logit_bias": "True",
        "semantic_bias_scale": "0.10",
        "semantic_profile_layers": "first",
    }),
    ("note_add_allmods_scale010", {
        "use_semantic_expert_profiles": "True",
        "use_semantic_logit_bias": "True",
        "semantic_bias_scale": "0.10",
        "semantic_profile_modalities": "ts,cxr,txt,ecg",
    }),
    ("note_add_allmods_scale020", {
        "use_semantic_expert_profiles": "True",
        "use_semantic_logit_bias": "True",
        "semantic_bias_scale": "0.20",
        "semantic_profile_modalities": "ts,cxr,txt,ecg",
    }),
    ("note_replace_txt_scale010", {
        "use_semantic_expert_profiles": "True",
        "semantic_only_router": "True",
        "semantic_bias_scale": "0.10",
        "semantic_profile_fusion": "replace",
    }),
    ("note_replace_allmods_scale010", {
        "use_semantic_expert_profiles": "True",
        "semantic_only_router": "True",
        "semantic_bias_scale": "0.10",
        "semantic_profile_fusion": "replace",
        "semantic_profile_modalities": "ts,cxr,txt,ecg",
    }),
    ("note_add_txt_scale010_shuffled", {
        "use_semantic_expert_profiles": "True",
        "use_semantic_logit_bias": "True",
        "semantic_bias_scale": "0.10",
        "semantic_profile_shuffle_assignments": "True",
    }),
    ("note_add_txt_scale010_randomctrl", {
        "use_semantic_expert_profiles": "True",
        "use_semantic_logit_bias": "True",
        "semantic_bias_scale": "0.10",
        "semantic_profile_set": "random_clinical_control",
    }),
]


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task, task_short, primary_metric, num_labels in TASKS:
        for seed in SEEDS:
            for variant_name, overrides in VARIANTS:
                row = base_row(task, task_short, primary_metric, num_labels, seed)
                row.update(overrides)
                row["experiment_group"] = "semantic_same_space_permod"
                row["variant_name"] = variant_name
                row["config_id"] = f"w33_semspace_{variant_name}_{task_short}_s{seed}"
                row["output_dir"] = str(output_dir(variant_name, task_short, seed) / row["config_id"])
                rows.append(row)
    return rows


def main() -> None:
    ensure_week33_dirs()
    rows = build_rows()
    manifest_path = MANIFEST_ROOT / "week33_semantic_same_space_manifest.tsv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(manifest_path)
    print(f"rows\t{len(rows)}")


if __name__ == "__main__":
    main()
