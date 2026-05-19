import argparse
import csv
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.model import default_expert_profiles
from core.train import ORGAN_KEYWORDS, EXPERT_PROFILE_NAMES
from utils.util import loadBert


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose note/profile cosine similarity before MoE routing.")
    parser.add_argument("--file_path", default="../../data/MIMIC-IV")
    parser.add_argument("--task", default="ihm-48-cxr-notes-ecg")
    parser.add_argument("--model_name", default="bioLongformer")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--num_of_notes", type=int, default=5)
    parser.add_argument("--notes_order", default="Last", choices=["First", "Last"])
    parser.add_argument("--semantic_profile_set", default="icu_organ_system")
    parser.add_argument("--note_pooling", default="max", choices=["max", "mean"])
    parser.add_argument("--output_dir", default="/home/pham156/MoE/FuseMoE_poly/out/Week_26/note_profile_similarity")
    parser.add_argument("--max_text_chars", type=int, default=1000)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def weak_organ_labels(note_texts):
    joined = " ".join(str(t) for t in note_texts).lower()
    labels = []
    for organ, keywords in ORGAN_KEYWORDS.items():
        if any(keyword in joined for keyword in keywords):
            labels.append(organ)
    return labels


def load_split(file_path, split, task):
    path = os.path.join(file_path, f"{split}_{task}_stays.pkl")
    print(f"Loading {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def select_notes_and_embeddings(example, num_notes, notes_order):
    if example.get("text_missing", False) or "text_data" not in example or "text_embeddings" not in example:
        return [], None

    note_texts = list(example["text_data"])
    embeddings = np.asarray(example["text_embeddings"], dtype=np.float32)
    if len(note_texts) == 0 or embeddings.size == 0:
        return [], None

    if notes_order == "Last":
        note_texts = note_texts[-num_notes:]
        embeddings = embeddings[-num_notes:]
    else:
        note_texts = note_texts[:num_notes]
        embeddings = embeddings[:num_notes]

    return note_texts, torch.tensor(embeddings, dtype=torch.float32)


def encode_profiles(args, device):
    biobert, _, tokenizer = loadBert(args, device)
    profiles = default_expert_profiles(args.semantic_profile_set)
    encoded = tokenizer(
        profiles,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        padding=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    biobert.eval()
    with torch.no_grad():
        outputs = biobert(**encoded)
        profile_embeddings = outputs[0][:, 0, :].detach().float().cpu()
    return F.normalize(profile_embeddings, dim=-1)


def score_example(note_embeddings, profile_embeddings, pooling):
    if note_embeddings is None or note_embeddings.numel() == 0:
        return torch.zeros(profile_embeddings.size(0), dtype=torch.float32)
    note_embeddings = F.normalize(note_embeddings.float(), dim=-1)
    note_scores = note_embeddings @ profile_embeddings.t()
    if pooling == "mean":
        return note_scores.mean(dim=0)
    return note_scores.max(dim=0).values


def summarize_rows(rows):
    total = len(rows)
    weak_counts = {name: 0 for name in EXPERT_PROFILE_NAMES}
    top1_counts = {name: 0 for name in EXPERT_PROFILE_NAMES}
    top2_counts = {name: 0 for name in EXPERT_PROFILE_NAMES}
    top1_match = {name: 0 for name in EXPERT_PROFILE_NAMES}
    top2_match = {name: 0 for name in EXPERT_PROFILE_NAMES}
    any_top2_match = 0

    for row in rows:
        weak = set(row["weak_organ_labels"].split("|")) if row["weak_organ_labels"] else set()
        top1 = row["top1_profile"]
        top2 = set(row["top2_profiles"].split("|")) if row["top2_profiles"] else set()
        for name in weak:
            if name in weak_counts:
                weak_counts[name] += 1
                top1_match[name] += int(top1 == name)
                top2_match[name] += int(name in top2)
        if weak & top2:
            any_top2_match += 1
        top1_counts[top1] += 1
        for name in top2:
            if name in top2_counts:
                top2_counts[name] += 1

    return {
        "total": total,
        "weak_counts": weak_counts,
        "top1_counts": top1_counts,
        "top2_counts": top2_counts,
        "top1_match": top1_match,
        "top2_match": top2_match,
        "any_top2_match": any_top2_match,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    profile_embeddings = encode_profiles(args, device)

    all_summary_rows = []
    for split in ["train", "val", "test"]:
        data = load_split(args.file_path, split, args.task)
        output_path = os.path.join(args.output_dir, f"{split}_note_profile_similarity.csv")
        rows = []

        with open(output_path, "w", newline="") as f:
            fieldnames = [
                "split",
                "sample_id",
                "label",
                "weak_organ_labels",
                "top1_profile",
                "top2_profiles",
                "score_cardiovascular",
                "score_respiratory",
                "score_renal_metabolic",
                "score_infection",
                "note_text_snippet",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for example in data:
                note_texts, note_embeddings = select_notes_and_embeddings(
                    example,
                    num_notes=args.num_of_notes,
                    notes_order=args.notes_order,
                )
                scores = score_example(note_embeddings, profile_embeddings, args.note_pooling)
                top_values, top_indices = torch.topk(scores, k=min(2, scores.numel()))
                top_names = [EXPERT_PROFILE_NAMES[int(i)] for i in top_indices.tolist()]
                labels = weak_organ_labels(note_texts)
                snippet = " ".join(str(t).replace("\n", " ") for t in note_texts)[: args.max_text_chars]
                row = {
                    "split": split,
                    "sample_id": example.get("name", ""),
                    "label": example.get("label", ""),
                    "weak_organ_labels": "|".join(labels),
                    "top1_profile": top_names[0] if top_names else "",
                    "top2_profiles": "|".join(top_names),
                    "score_cardiovascular": float(scores[0]),
                    "score_respiratory": float(scores[1]),
                    "score_renal_metabolic": float(scores[2]),
                    "score_infection": float(scores[3]),
                    "note_text_snippet": snippet,
                }
                writer.writerow(row)
                rows.append(row)

        summary = summarize_rows(rows)
        print(f"===== {split.upper()} SUMMARY =====")
        print(f"rows: {summary['total']}")
        print(f"weak_counts: {summary['weak_counts']}")
        print(f"top1_counts: {summary['top1_counts']}")
        print(f"top2_counts: {summary['top2_counts']}")
        print(f"top2_any_weak_overlap: {summary['any_top2_match']} / {summary['total']} = {summary['any_top2_match'] / max(summary['total'], 1):.4f}")
        for name in EXPERT_PROFILE_NAMES:
            denom = max(summary["weak_counts"][name], 1)
            print(
                f"{name}: top1_match={summary['top1_match'][name]}/{summary['weak_counts'][name]} "
                f"({summary['top1_match'][name] / denom:.4f}), "
                f"top2_match={summary['top2_match'][name]}/{summary['weak_counts'][name]} "
                f"({summary['top2_match'][name] / denom:.4f})"
            )
            all_summary_rows.append({
                "split": split,
                "profile": name,
                "total_rows": summary["total"],
                "weak_count": summary["weak_counts"][name],
                "top1_count": summary["top1_counts"][name],
                "top2_count": summary["top2_counts"][name],
                "top1_match": summary["top1_match"][name],
                "top2_match": summary["top2_match"][name],
            })

    summary_path = os.path.join(args.output_dir, "summary_note_profile_similarity.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "profile",
                "total_rows",
                "weak_count",
                "top1_count",
                "top2_count",
                "top1_match",
                "top2_match",
            ],
        )
        writer.writeheader()
        writer.writerows(all_summary_rows)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
