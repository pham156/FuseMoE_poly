#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from layout import MANIFEST_ROOT, TARGET_ROOT, REPORT_ROOT, ensure_week33_dirs, train_output_dir


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


def base_row(task: str, task_short: str, primary_metric: str, num_labels: int, architecture: str, seed: int) -> dict[str, str]:
    row = {
        "status": "ready",
        "skip_reason": "",
        "task": task,
        "task_short": task_short,
        "primary_metric": primary_metric,
        "num_labels": str(num_labels),
        "architecture": architecture,
        "seed": str(seed),
        "gating_function": "softmax",
        "poly_power": "",
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
        "semantic_profile_source": "patient",
        "semantic_profile_embedding_source": "biolongformer",
        "semantic_profile_shuffle_assignments": "False",
        "use_router_organ_supervision": "False",
        "router_organ_supervision_coef": "1.0",
        "router_organ_supervision_class_balanced": "False",
        "use_missing_modality_recon": "False",
        "missing_modality_recon_coef": "0.1",
        "missing_modality_recon_targets": "text,cxr,ecg",
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
    if architecture == "recon_ref":
        row["use_missing_modality_recon"] = "True"
        row["missing_modality_recon_targets"] = "cxr,ecg"
    return row


def semantic_router_rows() -> list[dict[str, str]]:
    rows = []
    variants = [
        ("baseline_learned_router", {"semantic_bias_scale": "0.0"}),
        ("semantic_bias_weak", {"use_semantic_logit_bias": "True", "semantic_bias_scale": "0.1"}),
        ("semantic_bias_medium", {"use_semantic_logit_bias": "True", "semantic_bias_scale": "0.5"}),
        ("semantic_bias_strong", {"use_semantic_logit_bias": "True", "semantic_bias_scale": "1.0"}),
        ("semantic_only_router", {"semantic_only_router": "True", "semantic_bias_scale": "1.0"}),
        ("shuffled_profile_control", {"use_semantic_logit_bias": "True", "semantic_bias_scale": "0.5", "semantic_profile_shuffle_assignments": "True"}),
        ("random_profile_control", {"use_semantic_logit_bias": "True", "semantic_bias_scale": "0.5", "semantic_profile_set": "random_clinical_control"}),
    ]
    for task, task_short, primary_metric, num_labels in TASKS:
        for seed in SEEDS:
            for variant_name, flags in variants:
                row = base_row(task, task_short, primary_metric, num_labels, "moe_ref", seed)
                row.update(flags)
                row["experiment_group"] = "semantic_router"
                row["variant_name"] = variant_name
                row["config_id"] = f"w33_semrouter_{variant_name}_moe_ref_{task_short}_s{seed}"
                row["output_dir"] = str(train_output_dir("semantic_router", variant_name, "moe_ref", task_short, seed) / row["config_id"])
                rows.append(row)
    return rows


def missing_recovery_rows() -> list[dict[str, str]]:
    rows = []
    variants = [
        ("zero_missing_baseline", {}),
        ("learned_missing_token", {"use_missing_modality_proxies": "True"}),
        ("cross_modal_recovery", {
            "use_missing_modality_proxies": "True",
            "use_cross_modal_missing_proxies": "True",
            "use_missing_modality_recon": "True",
            "train_cxr_modality_dropout": "0.15",
            "train_text_modality_dropout": "0.15",
            "train_ecg_modality_dropout": "0.15",
        }),
        ("recovery_regularized_router", {
            "use_missing_modality_recon": "True",
            "train_cxr_modality_dropout": "0.15",
            "train_text_modality_dropout": "0.15",
            "train_ecg_modality_dropout": "0.15",
        }),
    ]
    for task, task_short, primary_metric, num_labels in TASKS:
        for architecture in ("moe_ref", "recon_ref"):
            for seed in SEEDS:
                for variant_name, flags in variants:
                    row = base_row(task, task_short, primary_metric, num_labels, architecture, seed)
                    row.update(flags)
                    row["experiment_group"] = "missing_recovery"
                    row["variant_name"] = variant_name
                    row["config_id"] = f"w33_missrec_{variant_name}_{architecture}_{task_short}_s{seed}"
                    row["output_dir"] = str(train_output_dir("missing_recovery", variant_name, architecture, task_short, seed) / row["config_id"])
                    rows.append(row)
    return rows


def low_rank_rows() -> list[dict[str, str]]:
    rows = []
    variants = [
        ("mlp_expert_baseline", {"expert_type": "mlp", "gating_function": "poly", "poly_power": "4"}),
        ("low_rank_expert_r8", {"expert_type": "lora", "lora_rank": "8", "gating_function": "poly", "poly_power": "4"}),
        ("low_rank_expert_r16", {"expert_type": "lora", "lora_rank": "16", "gating_function": "poly", "poly_power": "4"}),
        ("shared_ffn_plus_low_rank_expert", {
            "expert_type": "residual_lora",
            "lora_rank": "8",
            "staged_shared_lora": "True",
            "staged_shared_lora_warmup_epochs": "4",
            "expert_orth_coef": "0.01",
            "gating_function": "poly",
            "poly_power": "4",
        }),
    ]
    task, task_short, primary_metric, num_labels = TASKS[0]
    for seed in SEEDS:
        for variant_name, flags in variants:
            row = base_row(task, task_short, primary_metric, num_labels, "moe_ref", seed)
            row.update(flags)
            row["experiment_group"] = "low_rank_experts"
            row["variant_name"] = variant_name
            row["config_id"] = f"w33_lowrank_{variant_name}_moe_ref_{task_short}_s{seed}"
            row["output_dir"] = str(train_output_dir("low_rank_experts", variant_name, "moe_ref", task_short, seed) / row["config_id"])
            rows.append(row)
    return rows


def cross_modal_init_rows() -> list[dict[str, str]]:
    rows = []
    target_template = str(TARGET_ROOT / "cross_modal_init" / "{split}_cross_modal_init_targets.csv")
    variants = [
        ("random", {"router_init": "normal"}),
        ("xavier", {"router_init": "xavier"}),
        ("semantic_warm_start", {
            "expert_init_strategy": "fixed_cohort",
            "expert_init_target_path": str(Path("/home/pham156/MoE/FuseMoE_poly/out/Week_29/semantic_init/targets/rule/{split}_rule_targets.csv")),
            "expert_init_epochs": "6",
            "expert_init_soft_targets": "True",
            "expert_init_freeze_router": "True",
            "expert_init_release_schedule": "linear",
        }),
        ("cross_modal_init", {
            "expert_init_strategy": "fixed_cohort",
            "expert_init_target_path": target_template,
            "expert_init_epochs": "6",
            "expert_init_soft_targets": "True",
            "expert_init_freeze_router": "True",
            "expert_init_release_schedule": "linear",
        }),
    ]
    for task, task_short, primary_metric, num_labels in TASKS[:1]:
        for seed in SEEDS:
            for variant_name, flags in variants:
                row = base_row(task, task_short, primary_metric, num_labels, "moe_ref", seed)
                row.update(flags)
                row["experiment_group"] = "cross_modal_init"
                row["variant_name"] = variant_name
                row["config_id"] = f"w33_xminit_{variant_name}_moe_ref_{task_short}_s{seed}"
                row["output_dir"] = str(train_output_dir("cross_modal_init", variant_name, "moe_ref", task_short, seed) / row["config_id"])
                rows.append(row)
    return rows


def main() -> None:
    ensure_week33_dirs()
    rows = semantic_router_rows() + missing_recovery_rows() + low_rank_rows() + cross_modal_init_rows()
    manifest_path = MANIFEST_ROOT / "week33_train_manifest.tsv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary_lines = [
        "# Week33 Proposal-Direct Train Manifest",
        "",
        f"- manifest: `{manifest_path}`",
        f"- rows: {len(rows)}",
        f"- semantic router rows: {sum(1 for row in rows if row['experiment_group'] == 'semantic_router')}",
        f"- missing recovery rows: {sum(1 for row in rows if row['experiment_group'] == 'missing_recovery')}",
        f"- low-rank rows: {sum(1 for row in rows if row['experiment_group'] == 'low_rank_experts')}",
        f"- cross-modal init rows: {sum(1 for row in rows if row['experiment_group'] == 'cross_modal_init')}",
    ]
    (REPORT_ROOT / "week33_train_manifest.md").write_text("\n".join(summary_lines) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
