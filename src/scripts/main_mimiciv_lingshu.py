import pandas as pd
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import wandb
import pickle
from torch.utils.data import Dataset, DataLoader, Subset, DistributedSampler, RandomSampler
import numpy as np


from tensorboardX import SummaryWriter

import warnings
import time
import sys
import logging
import os
logger = logging.getLogger(__name__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import *
from core.lingshu_pseudotoken import LingshuOrganLoraMoEModel, LingshuPseudoTokenModel
from core.train import *
from utils.checkpoint import *
from utils.util import *
from accelerate import Accelerator
from core.interp import *
from transformers import AutoTokenizer, BertTokenizer


class Struct(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


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
        # Stack each modality across batch
        num_mods = len(batch[0][0])
        mods_stacked = []
        for i in range(num_mods):
            mods_stacked.append(torch.stack([m[i] for m in mod_lists]))
        labels = torch.tensor(labels)
        if has_subject_ids:
            return mods_stacked, labels, torch.tensor(subject_ids)
        return mods_stacked, labels

    base_path = args.file_path  # note: use file_path, not data_dir
    
    def load_subject_data(subject_ids, split_name):
        merged = []
        for subject_id in subject_ids:
            subject_path = os.path.join(base_path, f"subject_{subject_id}.pkl")
            if not os.path.exists(subject_path):
                raise FileNotFoundError(
                    f"Missing PAM subject file for {split_name}: {subject_path}. "
                    "Re-run prepare_pam_data.py to generate per-subject files."
                )
            with open(subject_path, 'rb') as f:
                subject_samples = pickle.load(f)
                merged.extend((mods, label, subject_id) for mods, label in subject_samples)
        return merged

    def load_legacy_split(filename):
        with open(os.path.join(base_path, filename), 'rb') as f:
            return pickle.load(f)

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

    # Fit normalization only on the training split, then apply it to val/test.
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

    # Compute inverse-frequency class weights from the actual training set used.
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

def get_stratified_permutation(dataset, labels, rng):
    """
    Returns a permutation of dataset indices such that class proportions are preserved
    at every prefix. Works for binary or multiclass labels (0..C-1).
    """
    # Separate indices by class
    class_indices = {c: [] for c in set(labels)}
    for i, l in enumerate(labels):
        class_indices[l].append(i)
    # Shuffle each class
    for c in class_indices:
        class_indices[c] = rng.permutation(class_indices[c])
    # Interleave cyclically
    max_len = max(len(lst) for lst in class_indices.values())
    perm = []
    for i in range(max_len):
        for c in sorted(class_indices.keys()):  # deterministic order
            if i < len(class_indices[c]):
                perm.append(class_indices[c][i])
    # The list may be longer than the dataset if some classes are shorter; trim
    return perm[:len(dataset)]

def main():
    args = parse_args()

    if args.dataset == "pam":
        args.modeltype = "pam"         # to distinguish in logging, but not used
        # PAMAP2 setup in this code path uses 4 modalities: chest, hand, ankle, heart_rate.
        args.num_modalities = 4

    prefix = "fusemoe_new"

    if args.cross_method == 'hme':
        if 'poly' in args.gating_function:
            run_name = f"{prefix}_{args.gating_function[0]}_{args.gating_function[1]}_{args.poly_power}_{args.router_type}_hme_exp{args.num_of_experts}_k{args.top_k}"
        elif 'student_t' in args.gating_function:
            run_name = f"{prefix}_{args.gating_function[0]}_{args.gating_function[1]}_{args.student_degree}_{args.router_type}_hme_exp{args.num_of_experts}_k{args.top_k}"
        else:
            run_name = f"{prefix}_{args.gating_function[0]}_{args.gating_function[1]}_{args.router_type}_hme_exp{args.num_of_experts}_k{args.top_k}"

    elif args.cross_method == 'moe':
        if 'poly' in args.gating_function:
            run_name = f"{prefix}_{args.gating_function[0]}_{args.poly_power}_{args.router_type}_moe_exp{args.num_of_experts[0]}_k{args.top_k[0]}"
        elif 'student_t' in args.gating_function:
            run_name = f"{prefix}_{args.gating_function[0]}_{args.student_degree}_{args.router_type}_moe_exp{args.num_of_experts[0]}_k{args.top_k[0]}"
        else:
            run_name = f"{prefix}_{args.gating_function[0]}_{args.router_type}_moe_exp{args.num_of_experts[0]}_k{args.top_k[0]}"

    noise_tag = "noise" if args.noisy_gating else "clean"
    run_name = f"{run_name}_{noise_tag}"

    # wandb.init(
    #     project="polymoe_test",
    #     name=run_name,
    #     config=vars(args),
    #     dir="../../wandb"
    # )


    print(args)

    if args.fp16:
        args.mixed_precision="fp16"
    else:
        args.mixed_precision="no"
    accelerator = Accelerator(mixed_precision=args.mixed_precision,cpu=args.cpu)

    device = accelerator.device
    print(device)
    if not getattr(args, "disable_run_folder_save", False):
        os.makedirs(args.output_dir, exist_ok = True)
    if args.tensorboard_dir!=None:
        writer = SummaryWriter(args.tensorboard_dir)
    else:
        writer=None

    warnings.filterwarnings('ignore')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)
    print(args)
    make_save_dir(args)
    # exit()

    # if args.seed==0:
    #     copy_file(args.ck_file_path+'model/', src=os.getcwd())
    if args.mode=='train':
        if 'Text' in args.modeltype:
            tokenizer = load_tokenizer_only(args)
            BioBert = None
        else:
            BioBert, tokenizer = None, None

        if args.use_instruction_router:
            raise ValueError("main_mimiciv_lingshu.py does not support the old instruction router path.")
        else:
            args.router_instruction_dim = None

        from preprocessing.data_mimiciv import data_perpare

        # train_dataset, train_sampler, train_dataloader = data_perpare(args, 'train', tokenizer)
        # val_dataset, val_sampler, val_dataloader = data_perpare(args, 'val', tokenizer)
        # _, _, test_data_loader = data_perpare(args,'test',tokenizer)

        if args.dataset == "mimic":
            from preprocessing.data_mimiciv import data_perpare, TSNote_Irg, TextTSIrgcollate_fn
            from torch.utils.data import DataLoader, Subset
            import numpy as np

            # Create full training dataset
            train_dataset = TSNote_Irg(args, 'train', tokenizer)

            # Validation and test loaders (full)
            _, _, val_dataloader = data_perpare(args, 'val', tokenizer)
            _, _, test_data_loader = data_perpare(args, 'test', tokenizer)

            if args.train_sample_ratio < 1.0:
                total = len(train_dataset)
                num_samples = int(total * args.train_sample_ratio)
                rng = np.random.RandomState(args.seed)

                if args.task in ['ihm', 'los']:
                    # Binary tasks: stratify by label (0 or 1)
                    labels = [train_dataset[i]['label'].item() for i in range(total)]
                    perm = get_stratified_permutation(train_dataset, labels, rng)
                else:
                    # Multi‑label tasks (phenotyping): simple random permutation (nested)
                    perm = rng.permutation(total)

                indices = perm[:num_samples]
                train_subset = Subset(train_dataset, indices)
                train_dataloader = DataLoader(train_subset,
                                            batch_size=args.train_batch_size,
                                            shuffle=True,
                                            collate_fn=TextTSIrgcollate_fn,
                                            num_workers=0)
                print(f"Training with {len(train_subset)} samples (ratio={args.train_sample_ratio})")
            else:
                # Full dataset
                _, _, train_dataloader = data_perpare(args, 'train', tokenizer)
        elif args.dataset == "pam":
            train_dataloader, val_dataloader, test_data_loader, pam_class_weights, pam_modality_dims = get_pam_dataloaders(args)

        
    # if args.modeltype == 'Text':
    #     # pure text
    #     model= TextModel(args=args,device=device,orig_d_txt=768,Biobert=BioBert)
    # elif args.modeltype == 'TS':
    #     # pure time series
    #     model= TSMixed(args=args,device=device,orig_d_ts=30,orig_reg_d_ts=60, ts_seq_num=args.tt_max)
    # else:
    #     # multimodal fusion
    #     model= MULTCrossModel(args=args,device=device,orig_d_ts=30, orig_reg_d_ts=60, orig_d_txt=768,ts_seq_num=args.tt_max,text_seq_num=args.num_of_notes,Biobert=BioBert)
    # print(device)
    
    if args.dataset == "mimic":
        if args.lingshu_architecture == "organ_lora_moe":
            model = LingshuOrganLoraMoEModel(args=args, device=device)
        else:
            model = LingshuPseudoTokenModel(args=args, device=device)
    elif args.dataset == "pam":
        modality_dims = pam_modality_dims
        # Ensure cross_method is set (from args)
        from core.model import FlexiModalMULTCrossModel
        model = FlexiModalMULTCrossModel(args, device, modality_dims)
        if args.mode == "train":
            weight_tensor = torch.tensor(pam_class_weights, dtype=torch.float32, device=device)
            model.loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)

    if args.dataset == "mimic":
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=args.ts_learning_rate)
    elif args.modeltype == "pam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
    else:
        raise ValueError("Unknown modeltype in optimizer.")

    model, optimizer, train_dataloader,val_dataloader,test_data_loader = \
    accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, test_data_loader)

    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Model has {total_params:,} trainable parameters")
    
    best_model_state, best_val_score = trainer_irg(model=model,args=args,accelerator=accelerator,train_dataloader=train_dataloader,\
        dev_dataloader=val_dataloader, test_data_loader=test_data_loader, device=device,\
        optimizer=optimizer,writer=writer, dataset=args.dataset)
    # eval_test(args,model,test_data_loader, device)
    print(f"Best validation ({args.primary_metric}): {best_val_score}")
    model.load_state_dict(best_model_state)

    test_val = evaluate_irg(
        args=args,
        device=device,
        data_loader=test_data_loader,
        model=model,
        mode="test",
        dataset=args.dataset
    )
    print("\n===== FINAL TEST RESULTS =====")
    for k, v in test_val.items():
        # wandb.log({f"test/{k}": v})
        print(f"{k}: {v}")
    print("================================\n")

        

    print(f"New maximum memory allocated on GPU: {torch.cuda.max_memory_allocated(device)} bytes")
    if getattr(args, "disable_run_folder_save", False):
        print("Results not saved to run_folder (--disable_run_folder_save).")
    else:
        print(f'Results saved in:\n{args.ck_file_path}')


if __name__ == "__main__":

    import time
    start_time = time.time()
    main()
    # wandb.finish()
    print("--- %s seconds ---" % (time.time() - start_time))
