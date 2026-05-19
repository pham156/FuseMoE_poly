import argparse
import csv
import glob
import os
import re
from collections import defaultdict


EXPERTS = ["e0", "e1", "e2", "e3"]
ORGANS = ["cardiovascular", "respiratory", "renal_metabolic", "infection"]


def parse_args():
    parser = argparse.ArgumentParser(description="Post-hoc alignment between prototype gates and weak organ labels.")
    parser.add_argument(
        "--input_glob",
        default="/home/pham156/MoE/FuseMoE_poly/out/Week_26/prototype_router/*router_diagnostics.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/pham156/MoE/FuseMoE_poly/out/Week_26/prototype_router/alignment",
    )
    parser.add_argument("--layers", nargs="*", type=int, default=[0, 1, 2])
    return parser.parse_args()


def parse_ratio(path):
    match = re.search(r"ratio_(\dp\d+)", os.path.basename(path))
    return match.group(1) if match else "unknown"


def parse_gate_weights(value, num_experts=4):
    weights = [0.0] * num_experts
    for item in str(value).split("|"):
        if not item:
            continue
        expert_id, weight = item.split(":")
        weights[int(expert_id)] = float(weight)
    return weights


def weak_labels(row):
    labels = row.get("weak_organ_labels", "")
    return set(label for label in labels.split("|") if label)


def mean(values):
    return sum(values) / max(len(values), 1)


def summarize_path(path, layers):
    ratio = parse_ratio(path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    summaries = []
    for layer in layers:
        gate_col = f"layer{layer}_topk_expert_ids_weights"
        if gate_col not in rows[0]:
            continue

        all_gates = [parse_gate_weights(row[gate_col]) for row in rows]
        overall = [mean([g[i] for g in all_gates]) for i in range(4)]

        for organ in ORGANS:
            pos = []
            neg = []
            for row, gates in zip(rows, all_gates):
                if organ in weak_labels(row):
                    pos.append(gates)
                else:
                    neg.append(gates)

            pos_mean = [mean([g[i] for g in pos]) for i in range(4)]
            neg_mean = [mean([g[i] for g in neg]) for i in range(4)]
            diff = [pos_mean[i] - neg_mean[i] for i in range(4)]
            top_expert = max(range(4), key=lambda i: pos_mean[i])
            top_lift = max(range(4), key=lambda i: diff[i])

            summaries.append({
                "ratio": ratio,
                "layer": layer,
                "weak_label": organ,
                "count_pos": len(pos),
                "count_neg": len(neg),
                "top_avg_expert": f"e{top_expert}",
                "top_lift_expert": f"e{top_lift}",
                "overall_e0": overall[0],
                "overall_e1": overall[1],
                "overall_e2": overall[2],
                "overall_e3": overall[3],
                "pos_avg_e0": pos_mean[0],
                "pos_avg_e1": pos_mean[1],
                "pos_avg_e2": pos_mean[2],
                "pos_avg_e3": pos_mean[3],
                "neg_avg_e0": neg_mean[0],
                "neg_avg_e1": neg_mean[1],
                "neg_avg_e2": neg_mean[2],
                "neg_avg_e3": neg_mean[3],
                "lift_e0": diff[0],
                "lift_e1": diff[1],
                "lift_e2": diff[2],
                "lift_e3": diff[3],
            })

    return summaries


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, rows):
    with open(path, "w") as f:
        for ratio in sorted(set(row["ratio"] for row in rows)):
            f.write(f"===== ratio {ratio} =====\n")
            for layer in sorted(set(row["layer"] for row in rows if row["ratio"] == ratio)):
                f.write(f"layer {layer}\n")
                subset = [row for row in rows if row["ratio"] == ratio and row["layer"] == layer]
                for row in subset:
                    lifts = [float(row[f"lift_e{i}"]) for i in range(4)]
                    pos = [float(row[f"pos_avg_e{i}"]) for i in range(4)]
                    f.write(
                        f"  {row['weak_label']:15s} n={row['count_pos']:4d} "
                        f"top_avg={row['top_avg_expert']} "
                        f"top_lift={row['top_lift_expert']} "
                        f"pos=[{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f},{pos[3]:.4f}] "
                        f"lift=[{lifts[0]:+.4f},{lifts[1]:+.4f},{lifts[2]:+.4f},{lifts[3]:+.4f}]\n"
                    )
                f.write("\n")


def main():
    args = parse_args()
    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.input_glob}")

    all_rows = []
    for path in paths:
        all_rows.extend(summarize_path(path, args.layers))

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "prototype_weak_label_alignment_summary.csv")
    report_path = os.path.join(args.output_dir, "prototype_weak_label_alignment_report.txt")
    write_csv(summary_path, all_rows)
    write_report(report_path, all_rows)
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
