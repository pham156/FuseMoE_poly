import argparse
import csv
import os
import pickle
import re
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.model import default_expert_profiles
from core.train import EXPERT_PROFILE_NAMES, ORGAN_KEYWORDS
from utils.util import loadBert


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose sentence-level note/profile cosine similarity.")
    parser.add_argument("--file_path", default="/home/pham156/MoE/FuseMoE_poly/data/MIMIC-IV")
    parser.add_argument("--task", default="ihm-48-cxr-notes-ecg")
    parser.add_argument("--model_name", default="bioLongformer")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--pad_to_max_length", action="store_true")
    parser.add_argument("--num_of_notes", type=int, default=5)
    parser.add_argument("--notes_order", default="Last", choices=["First", "Last"])
    parser.add_argument("--semantic_profile_set", default="icu_organ_system")
    parser.add_argument("--output_dir", default="/home/pham156/MoE/FuseMoE_poly/out/Week_26/sentence_profile_similarity")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_sentence_chars", type=int, default=500)
    parser.add_argument("--min_sentence_chars", type=int, default=20)
    parser.add_argument("--max_sentences_per_note", type=int, default=80)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_split(file_path, split, task):
    path = os.path.join(file_path, f"{split}_{task}_stays.pkl")
    print(f"Loading {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def select_notes(example, num_notes, notes_order):
    if example.get("text_missing", False) or "text_data" not in example:
        return []

    note_texts = list(example["text_data"])
    if not note_texts:
        return []

    if notes_order == "Last":
        return note_texts[-num_notes:]
    return note_texts[:num_notes]


def split_sentences(text, min_chars, max_chars, max_sentences):
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []

    pieces = re.split(r"(?<=[.!?])\s+|(?:\s{2,})|(?:\n+)", text)
    sentences = []
    for piece in pieces:
        piece = piece.strip()
        if len(piece) < min_chars:
            continue
        if len(piece) > max_chars:
            for start in range(0, len(piece), max_chars):
                chunk = piece[start : start + max_chars].strip()
                if len(chunk) >= min_chars:
                    sentences.append(chunk)
        else:
            sentences.append(piece)
        if len(sentences) >= max_sentences:
            break
    return sentences


def weak_organ_labels(text):
    lowered = str(text).lower()
    labels = []
    for organ, keywords in ORGAN_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            labels.append(organ)
    return labels


def encode_texts(model, tokenizer, texts, args, device):
    if not texts:
        return torch.empty(0, len(EXPERT_PROFILE_NAMES))

    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                padding=True,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            embeddings.append(outputs[0][:, 0, :].detach().float().cpu())
    return torch.cat(embeddings, dim=0)


def encode_profiles(model, tokenizer, args, device):
    profiles = default_expert_profiles(args.semantic_profile_set)
    profile_embeddings = encode_texts(model, tokenizer, profiles, args, device)
    return F.normalize(profile_embeddings, dim=-1)


def top_names(scores):
    values, indices = torch.topk(scores, k=min(2, scores.numel()))
    names = [EXPERT_PROFILE_NAMES[int(i)] for i in indices.tolist()]
    return names, values.tolist()


def write_split(split, data, model, tokenizer, profile_embeddings, args, device):
    sentence_path = os.path.join(args.output_dir, f"{split}_sentence_profile_similarity.csv")
    aggregate_path = os.path.join(args.output_dir, f"{split}_sentence_aggregate_profile_similarity.csv")

    sentence_fields = [
        "split",
        "sample_id",
        "label",
        "note_index",
        "sentence_index",
        "weak_organ_labels",
        "top1_profile",
        "top2_profiles",
        "score_cardiovascular",
        "score_respiratory",
        "score_renal_metabolic",
        "score_infection",
        "sentence_text",
    ]
    aggregate_fields = [
        "split",
        "sample_id",
        "label",
        "num_sentences",
        "weak_organ_labels",
        "top1_profile",
        "top2_profiles",
        "score_cardiovascular",
        "score_respiratory",
        "score_renal_metabolic",
        "score_infection",
    ]

    top1_counts = {name: 0 for name in EXPERT_PROFILE_NAMES}
    top2_counts = {name: 0 for name in EXPERT_PROFILE_NAMES}
    sentence_count = 0

    with open(sentence_path, "w", newline="") as sf, open(aggregate_path, "w", newline="") as af:
        sentence_writer = csv.DictWriter(sf, fieldnames=sentence_fields)
        aggregate_writer = csv.DictWriter(af, fieldnames=aggregate_fields)
        sentence_writer.writeheader()
        aggregate_writer.writeheader()

        for example_idx, example in enumerate(data):
            sample_id = example.get("name", example_idx)
            label = example.get("label", "")
            notes = select_notes(example, args.num_of_notes, args.notes_order)

            sentence_records = []
            for note_idx, note in enumerate(notes):
                sentences = split_sentences(
                    note,
                    min_chars=args.min_sentence_chars,
                    max_chars=args.max_sentence_chars,
                    max_sentences=args.max_sentences_per_note,
                )
                for sent_idx, sentence in enumerate(sentences):
                    sentence_records.append((note_idx, sent_idx, sentence))

            if not sentence_records:
                scores = torch.zeros(len(EXPERT_PROFILE_NAMES), dtype=torch.float32)
                names, _ = top_names(scores)
                aggregate_writer.writerow({
                    "split": split,
                    "sample_id": sample_id,
                    "label": label,
                    "num_sentences": 0,
                    "weak_organ_labels": "",
                    "top1_profile": names[0],
                    "top2_profiles": "|".join(names),
                    "score_cardiovascular": float(scores[0]),
                    "score_respiratory": float(scores[1]),
                    "score_renal_metabolic": float(scores[2]),
                    "score_infection": float(scores[3]),
                })
                continue

            texts = [record[2] for record in sentence_records]
            embeddings = encode_texts(model, tokenizer, texts, args, device)
            embeddings = F.normalize(embeddings, dim=-1)
            sentence_scores = embeddings @ profile_embeddings.t()
            aggregate_scores = sentence_scores.max(dim=0).values
            aggregate_names, _ = top_names(aggregate_scores)
            joined_labels = sorted(set(label for text in texts for label in weak_organ_labels(text)))

            aggregate_writer.writerow({
                "split": split,
                "sample_id": sample_id,
                "label": label,
                "num_sentences": len(texts),
                "weak_organ_labels": "|".join(joined_labels),
                "top1_profile": aggregate_names[0],
                "top2_profiles": "|".join(aggregate_names),
                "score_cardiovascular": float(aggregate_scores[0]),
                "score_respiratory": float(aggregate_scores[1]),
                "score_renal_metabolic": float(aggregate_scores[2]),
                "score_infection": float(aggregate_scores[3]),
            })

            for (note_idx, sent_idx, sentence), scores in zip(sentence_records, sentence_scores):
                names, _ = top_names(scores)
                top1_counts[names[0]] += 1
                for name in names:
                    top2_counts[name] += 1
                sentence_count += 1
                sentence_writer.writerow({
                    "split": split,
                    "sample_id": sample_id,
                    "label": label,
                    "note_index": note_idx,
                    "sentence_index": sent_idx,
                    "weak_organ_labels": "|".join(weak_organ_labels(sentence)),
                    "top1_profile": names[0],
                    "top2_profiles": "|".join(names),
                    "score_cardiovascular": float(scores[0]),
                    "score_respiratory": float(scores[1]),
                    "score_renal_metabolic": float(scores[2]),
                    "score_infection": float(scores[3]),
                    "sentence_text": sentence,
                })

    print(f"===== {split.upper()} SENTENCE SUMMARY =====")
    print(f"sentences: {sentence_count}")
    print(f"top1_counts: {top1_counts}")
    print(f"top2_counts: {top2_counts}")
    print(f"Wrote: {sentence_path}")
    print(f"Wrote: {aggregate_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, _, tokenizer = loadBert(args, device)
    profile_embeddings = encode_profiles(model, tokenizer, args, device)

    for split in ["train", "val", "test"]:
        data = load_split(args.file_path, split, args.task)
        write_split(split, data, model, tokenizer, profile_embeddings, args, device)


if __name__ == "__main__":
    main()
