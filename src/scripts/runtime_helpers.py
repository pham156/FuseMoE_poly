import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, BertTokenizer


def load_tokenizer_only(args):
    if args.model_name == "BioBert":
        return AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    if args.model_name == "bioRoberta":
        return AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
    if args.model_name == "Bert":
        return BertTokenizer.from_pretrained("bert-base-uncased")
    if args.model_name == "bioLongformer":
        tok_path = os.getenv("CLIN_LONGFORMER_DIR", "yikuan8/Clinical-Longformer")
        return AutoTokenizer.from_pretrained(tok_path)
    if args.model_path is not None:
        return AutoTokenizer.from_pretrained(args.model_path)
    raise ValueError("provide either a supported --model_name or --model_path for tokenization")


def get_pam_dataloaders(args):
    class PAMDataset(Dataset):
        def __init__(self, data, seq_len):
            self.data = data
            self.seq_len = seq_len

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            if len(item) == 3:
                modality_list, label, subject_id = item
                return modality_list, label, subject_id
            modality_list, label = item
            return modality_list, label

    def collate_fn(batch):
        has_subject_ids = len(batch[0]) == 3
        if has_subject_ids:
            mod_lists, labels, subject_ids = zip(*batch)
        else:
            mod_lists, labels = zip(*batch)
        num_mods = len(batch[0][0])
        mods_stacked = []
        for i in range(num_mods):
            mods_stacked.append(torch.stack([m[i] for m in mod_lists]))
        labels = torch.tensor(labels)
        if has_subject_ids:
            return mods_stacked, labels, torch.tensor(subject_ids)
        return mods_stacked, labels

    base_path = args.file_path

    def load_subject_data(subject_ids, split_name):
        merged = []
        for subject_id in subject_ids:
            subject_path = os.path.join(base_path, f"subject_{subject_id}.pkl")
            if not os.path.exists(subject_path):
                raise FileNotFoundError(
                    f"Missing PAM subject file for {split_name}: {subject_path}. "
                    "Re-run prepare_pam_data.py to generate per-subject files."
                )
            with open(subject_path, "rb") as handle:
                subject_samples = pickle.load(handle)
                merged.extend((mods, label, subject_id) for mods, label in subject_samples)
        return merged

    def load_legacy_split(filename):
        with open(os.path.join(base_path, filename), "rb") as handle:
            return pickle.load(handle)

    def compute_modality_stats(samples):
        stats = []
        num_mods = len(samples[0][0])
        for mod_idx in range(num_mods):
            flattened = torch.cat(
                [sample[0][mod_idx].reshape(-1, sample[0][mod_idx].shape[-1]) for sample in samples],
                dim=0,
            )
            mean = flattened.mean(dim=0)
            std = flattened.std(dim=0)
            std = torch.where(std < 1e-6, torch.ones_like(std), std)
            stats.append((mean, std))
        return stats

    def normalize_samples(samples, stats):
        normalized = []
        for sample in samples:
            mods, label = sample[0], sample[1]
            norm_mods = []
            for mod, (mean, std) in zip(mods, stats):
                norm_mods.append((mod - mean) / std)
            if len(sample) > 2:
                normalized.append((norm_mods, label, sample[2]))
            else:
                normalized.append((norm_mods, label))
        return normalized

    requested_subjects = sorted(set(args.pam_train_subjects + args.pam_val_subjects + args.pam_test_subjects))
    existing_subjects = [
        subject_id for subject_id in requested_subjects
        if os.path.exists(os.path.join(base_path, f"subject_{subject_id}.pkl"))
    ]
    missing_subjects = [subject_id for subject_id in requested_subjects if subject_id not in existing_subjects]

    modality_dims = None
    if len(existing_subjects) == len(requested_subjects):
        train_data = load_subject_data(args.pam_train_subjects, "train")
        val_data = load_subject_data(args.pam_val_subjects, "val")
        test_data = load_subject_data(args.pam_test_subjects, "test")
        print(f"PAM subject splits - train: {args.pam_train_subjects}, val: {args.pam_val_subjects}, test: {args.pam_test_subjects}")
    elif len(existing_subjects) > 0:
        raise FileNotFoundError(
            "Found some PAM per-subject files but not all requested ones. "
            f"Existing subjects: {existing_subjects}. Missing subjects: {missing_subjects}. "
            "Re-run prepare_pam_data.py so all requested subjects are generated consistently."
        )
    else:
        print("Per-subject PAM files not found, falling back to legacy aggregate split files.")
        train_data = load_legacy_split("train_all_subjects.pkl")
        val_data = load_legacy_split("val_all_subjects.pkl")
        test_data = load_legacy_split("test_all_subjects.pkl")

    if len(train_data) == 0:
        raise ValueError("PAM training split is empty. Check the selected train subjects and processed subject files.")

    modality_dims = [sample.shape[-1] for sample in train_data[0][0]]
    print(f"PAM modality dims inferred from processed data: {modality_dims}")

    pam_stats = compute_modality_stats(train_data)
    train_data = normalize_samples(train_data, pam_stats)
    val_data = normalize_samples(val_data, pam_stats)
    test_data = normalize_samples(test_data, pam_stats)

    train_dataset = PAMDataset(train_data, args.tt_max)
    if args.train_sample_ratio < 1.0:
        total = len(train_dataset)
        num = int(total * args.train_sample_ratio)
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(total, num, replace=False)
        train_dataset = Subset(train_dataset, indices)

    if isinstance(train_dataset, Subset):
        train_labels = [train_dataset.dataset.data[i][1] for i in train_dataset.indices]
    else:
        train_labels = [sample[1] for sample in train_dataset.data]
    class_counts = np.bincount(np.array(train_labels, dtype=np.int64), minlength=args.num_labels)
    class_counts = np.maximum(class_counts, 1)
    inv_freq = 1.0 / class_counts
    class_weights = inv_freq / inv_freq.sum() * len(inv_freq)

    pam_loader_workers = min(4, os.cpu_count() or 1)
    pam_loader_kwargs = {
        "num_workers": pam_loader_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if pam_loader_workers > 0:
        pam_loader_kwargs["persistent_workers"] = True
        pam_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **pam_loader_kwargs,
    )
    val_loader = DataLoader(
        PAMDataset(val_data, args.tt_max),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **pam_loader_kwargs,
    )
    test_loader = DataLoader(
        PAMDataset(test_data, args.tt_max),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **pam_loader_kwargs,
    )
    return train_loader, val_loader, test_loader, class_weights, modality_dims
