#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from common import read_tsv
from layout import MANIFEST_ROOT, REPORT_ROOT, ensure_week33_dirs, masked_eval_output_dir


MASKS = [
    ("full", ""),
    ("no_text", "text"),
    ("no_cxr", "cxr"),
    ("no_ecg", "ecg"),
    ("no_text_no_cxr", "text,cxr"),
    ("only_ts", "text,cxr,ecg"),
]


FIELDNAMES = [
    "status",
    "skip_reason",
    "config_id",
    "train_config_id",
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
    "checkpoint_path",
    "mask_name",
    "eval_force_missing_modalities",
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
    "disable_run_folder_save",
    "output_dir",
]


def main() -> None:
    ensure_week33_dirs()
    train_manifest = read_tsv(MANIFEST_ROOT / "week33_train_manifest.tsv")
    rows = []
    for train_row in train_manifest:
        if train_row.get("status") != "ready":
            continue
        if train_row.get("experiment_group") != "missing_recovery":
            continue
        checkpoint_path = str(Path(train_row["output_dir"]) / "best_model.pth.tar")
        for mask_name, forced in MASKS:
            row = {key: train_row.get(key, "") for key in FIELDNAMES if key in train_row}
            row["status"] = "ready"
            row["skip_reason"] = ""
            row["train_config_id"] = train_row["config_id"]
            row["config_id"] = f"{train_row['config_id']}__{mask_name}"
            row["checkpoint_path"] = checkpoint_path
            row["mask_name"] = mask_name
            row["eval_force_missing_modalities"] = forced
            row["output_dir"] = str(
                masked_eval_output_dir(
                    train_row["experiment_group"],
                    train_row["variant_name"],
                    train_row["architecture"],
                    train_row["task_short"],
                    int(train_row["seed"]),
                    mask_name,
                )
                / row["config_id"]
            )
            rows.append(row)

    manifest_path = MANIFEST_ROOT / "week33_masked_eval_manifest.tsv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    (REPORT_ROOT / "week33_masked_eval_manifest.md").write_text(
        "\n".join(
            [
                "# Week33 Missing-Recovery Masked Eval Manifest",
                "",
                f"- manifest: `{manifest_path}`",
                f"- rows: {len(rows)}",
            ]
        )
        + "\n"
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

