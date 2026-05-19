#!/usr/bin/env python3
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def topk_contains_profile(topk_profile_names, weak_label):
    if not topk_profile_names:
        return False
    groups = []
    for modality_group in topk_profile_names.split(";"):
        if "=" in modality_group:
            modality_group = modality_group.split("=", 1)[1]
        groups.extend([x for x in modality_group.split("|") if x])
    return weak_label in groups


def main():
    parser = argparse.ArgumentParser(description="Summarize router_diagnostics_test.csv files.")
    parser.add_argument("paths", nargs="+", help="CSV files or directories containing router_diagnostics_test.csv files.")
    args = parser.parse_args()

    csv_paths = []
    for raw_path in args.paths:
        path = Path(raw_path)
        if path.is_dir():
            csv_paths.extend(sorted(path.rglob("router_diagnostics_test.csv")))
        elif path.is_file():
            csv_paths.append(path)

    if not csv_paths:
        raise SystemExit("No router_diagnostics_test.csv files found.")

    total_rows = 0
    weak_counts = Counter()
    consistency_counts = defaultdict(lambda: Counter())

    for path in csv_paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                weak_labels = [x for x in row.get("weak_organ_labels", "").split("|") if x]
                for label in weak_labels:
                    weak_counts[label] += 1
                    consistency_counts[label]["matched"] += int(
                        topk_contains_profile(row.get("topk_expert_profile_names", ""), label)
                    )
                    consistency_counts[label]["total"] += 1

    print(f"files: {len(csv_paths)}")
    print(f"rows: {total_rows}")
    print()
    print("weak_label,count,topk_match,consistency")
    for label in sorted(weak_counts):
        total = consistency_counts[label]["total"]
        matched = consistency_counts[label]["matched"]
        consistency = matched / total if total else 0.0
        print(f"{label},{total},{matched},{consistency:.4f}")


if __name__ == "__main__":
    main()
