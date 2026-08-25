#!/usr/bin/env python3
import csv
from pathlib import Path

from layout import MANIFEST_ROOT, REPORT_ROOT, RESULTS_SCRATCH_ROOT, ensure_week31_dirs


TASKS = [
    ("pheno-all-cxr-notes-ecg", "macro_f1", 25, "pheno"),
    ("ihm-48-cxr-notes-ecg", "f1", 2, "ihm"),
    ("los-48-cxr-notes-ecg", "f1", 2, "los"),
]
ARCHITECTURES = ["moe_ref", "recon_ref"]
SEEDS = [32, 42, 52]
GATING_VARIANTS = [
    ("softmax", "", "softmax"),
    ("laplace", "", "laplace"),
    ("poly", "4", "poly4"),
]


FIELDNAMES = [
    "config_id",
    "status",
    "skip_reason",
    "experiment_group",
    "variant_name",
    "task",
    "task_short",
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
    "use_task_specific_router_heads",
    "multitask_shared_moe_trunk",
    "use_modality_mask_condition_router",
    "mask_condition_stats",
    "use_interaction_router",
    "interaction_router_only",
    "interaction_router_scale",
    "shared_semantic_memory_mode",
    "shared_semantic_memory_slots",
    "shared_semantic_memory_heads",
    "use_missing_modality_proxies",
    "use_cross_modal_missing_proxies",
    "cross_modal_proxy_hidden",
    "cross_modal_proxy_dropout",
    "missing_modality_proxy_init_std",
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


def _base_row(task, task_short, primary_metric, num_labels, architecture, seed):
    return {
        "status": "ready",
        "skip_reason": "",
        "task": task,
        "task_short": task_short,
        "primary_metric": primary_metric,
        "num_labels": num_labels,
        "architecture": architecture,
        "seed": seed,
        "modeltype": "TS_CXR_Text_ECG",
        "num_modalities": 4,
        "gating_function": "softmax",
        "poly_power": "",
        "router_type": "permod",
        "router_init": "normal",
        "router_init_std": "0.02",
        "use_task_condition_router": "False",
        "task_condition_stats": "False",
        "use_task_specific_router_heads": "False",
        "multitask_shared_moe_trunk": "False",
        "use_modality_mask_condition_router": "False",
        "mask_condition_stats": "False",
        "use_interaction_router": "False",
        "interaction_router_only": "False",
        "interaction_router_scale": "1.0",
        "shared_semantic_memory_mode": "none",
        "shared_semantic_memory_slots": "8",
        "shared_semantic_memory_heads": "1",
        "use_missing_modality_proxies": "False",
        "use_cross_modal_missing_proxies": "False",
        "cross_modal_proxy_hidden": "256",
        "cross_modal_proxy_dropout": "0.1",
        "missing_modality_proxy_init_std": "0.02",
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
        "log_interaction_features": "False",
    }


def _output_dir(group, architecture, variant_name, task_short, seed):
    return (
        RESULTS_SCRATCH_ROOT
        / "targeted_followup"
        / group
        / architecture
        / variant_name
        / task_short
        / f"seed_{seed}"
    )


def build_rows():
    for task, primary_metric, num_labels, task_short in TASKS:
        for architecture in ARCHITECTURES:
            for seed in SEEDS:
                for gating_function, poly_power, gating_label in GATING_VARIANTS:
                    for variant_name, flags in [
                        ("zero_missing_baseline", {"router_family": "permod_router", "init_strategy": "random"}),
                        ("learned_proxy_tokens", {
                            "router_family": "permod_router",
                            "init_strategy": "random",
                            "use_missing_modality_proxies": "True",
                        }),
                        ("cross_modal_proxy", {
                            "router_family": "permod_router",
                            "init_strategy": "random",
                            "use_cross_modal_missing_proxies": "True",
                        }),
                    ]:
                        row = _base_row(task, task_short, primary_metric, num_labels, architecture, seed)
                        row.update(flags)
                        row["gating_function"] = gating_function
                        row["poly_power"] = poly_power
                        row["experiment_group"] = "proxy"
                        row["variant_name"] = variant_name
                        row["config_id"] = f"w31follow_proxy_{variant_name}_{architecture}_{task_short}_{gating_label}_s{seed}"
                        row["week31_variant_id"] = f"proxy__{variant_name}__{gating_label}"
                        row["week31_router_family"] = row["router_family"]
                        row["output_dir"] = str(_output_dir("proxy", architecture, variant_name, task_short, seed) / row["config_id"])
                        yield row

                    for variant_name, flags in [
                        ("baseline_router", {"router_family": "permod_router", "init_strategy": "random"}),
                        ("dependency_router", {
                            "router_family": "interaction_router",
                            "init_strategy": "random",
                            "use_interaction_router": "True",
                            "log_interaction_features": "True",
                        }),
                        ("dependency_only_router", {
                            "router_family": "interaction_only_router",
                            "init_strategy": "random",
                            "use_interaction_router": "True",
                            "interaction_router_only": "True",
                            "log_interaction_features": "True",
                        }),
                    ]:
                        row = _base_row(task, task_short, primary_metric, num_labels, architecture, seed)
                        row.update(flags)
                        row["gating_function"] = gating_function
                        row["poly_power"] = poly_power
                        row["experiment_group"] = "dependency_router"
                        row["variant_name"] = variant_name
                        row["config_id"] = f"w31follow_dep_{variant_name}_{architecture}_{task_short}_{gating_label}_s{seed}"
                        row["week31_variant_id"] = f"dependency__{variant_name}__{gating_label}"
                        row["week31_router_family"] = row["router_family"]
                        row["output_dir"] = str(_output_dir("dependency_router", architecture, variant_name, task_short, seed) / row["config_id"])
                        yield row

    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            for gating_function, poly_power, gating_label in GATING_VARIANTS:
                for variant_name, flags in [
                    ("task_embedding_router", {
                        "router_family": "task_router",
                        "init_strategy": "random",
                        "use_task_condition_router": "True",
                        "task_condition_stats": "True",
                        "multitask_shared_moe_trunk": "True",
                    }),
                    ("per_task_router_heads", {
                        "router_family": "task_head_router",
                        "init_strategy": "random",
                        "use_task_specific_router_heads": "True",
                        "multitask_shared_moe_trunk": "True",
                    }),
                ]:
                    row = _base_row("pheno-all-cxr-notes-ecg", "multitask", "macro_f1", 25, architecture, seed)
                    row.update(flags)
                    row["gating_function"] = gating_function
                    row["poly_power"] = poly_power
                    row["experiment_group"] = "task_head"
                    row["variant_name"] = variant_name
                    row["config_id"] = f"w31follow_task_{variant_name}_{architecture}_{gating_label}_s{seed}"
                    row["week31_variant_id"] = f"task_head__{variant_name}__{gating_label}"
                    row["week31_router_family"] = row["router_family"]
                    row["output_dir"] = str(_output_dir("task_head", architecture, variant_name, "multitask", seed) / row["config_id"])
                    yield row


def main():
    ensure_week31_dirs()
    manifest_path = MANIFEST_ROOT / "week31_targeted_followup_manifest.tsv"
    rows = list(build_rows())
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_path = REPORT_ROOT / "week31_targeted_followup_manifest.md"
    lines = [
        "# Week31 Targeted Follow-up Manifest",
        "",
        f"- manifest: `{manifest_path}`",
        f"- rows: {len(rows)}",
        f"- proxy rows: {sum(1 for r in rows if r['experiment_group'] == 'proxy')}",
        f"- dependency rows: {sum(1 for r in rows if r['experiment_group'] == 'dependency_router')}",
        f"- task-head rows: {sum(1 for r in rows if r['experiment_group'] == 'task_head')}",
    ]
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
