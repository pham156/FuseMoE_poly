#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from common import parse_final_metrics, parse_router_summary, read_tsv, write_csv
from layout import AGGREGATE_ROOT, LOG_ROOT, MANIFEST_ROOT, REPORT_ROOT, ensure_week33_dirs


def main() -> None:
    ensure_week33_dirs()
    rows = []
    for manifest_name in ("week33_train_manifest.tsv", "week33_masked_eval_manifest.tsv"):
        manifest_path = MANIFEST_ROOT / manifest_name
        if not manifest_path.exists():
            continue
        for row in read_tsv(manifest_path):
            if row.get("status") != "ready":
                continue
            log_candidates = list(LOG_ROOT.glob(f"*_{row['config_id']}.out"))
            if not log_candidates:
                continue
            log_path = max(log_candidates, key=lambda path: path.stat().st_mtime)
            metrics = parse_final_metrics(log_path)
            router = parse_router_summary(log_path)
            rows.append(
                {
                    "manifest": manifest_name,
                    "config_id": row["config_id"],
                    "experiment_group": row.get("experiment_group", ""),
                    "variant_name": row.get("variant_name", ""),
                    "task_short": row.get("task_short", ""),
                    "architecture": row.get("architecture", ""),
                    "seed": row.get("seed", ""),
                    "mask_name": row.get("mask_name", ""),
                    "output_dir": row.get("output_dir", ""),
                    **metrics,
                    **router,
                }
            )
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else ["config_id"]
    write_csv(AGGREGATE_ROOT / "week33_results.csv", rows, fieldnames)
    (REPORT_ROOT / "week33_results_summary.md").write_text(
        "\n".join(
            [
                "# Week33 Parsed Results",
                "",
                f"- parsed rows: {len(rows)}",
                f"- output: `{AGGREGATE_ROOT / 'week33_results.csv'}`",
            ]
        ) + "\n"
    )


if __name__ == "__main__":
    main()

