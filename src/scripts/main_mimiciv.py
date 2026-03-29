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
from core.train import *
from utils.checkpoint import *
from utils.util import *
from accelerate import Accelerator
from core.interp import *


class Struct(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_pam_dataloaders(args):
    class PAMDataset(Dataset):
        def __init__(self, pkl_path, seq_len):
            with open(pkl_path, 'rb') as f:
                self.data = pickle.load(f)
            self.seq_len = seq_len

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            modality_list, label = self.data[idx]   # list of tensors, label
            # For simplicity, we don't need masks/timestamps – the model will use linear projections.
            # But if you want to keep compatibility, you can add dummy masks and timestamps.
            # We'll return the list and the label.
            return modality_list, label
    

    def collate_fn(batch):
        mod_lists, labels = zip(*batch)
        # Stack each modality across batch
        num_mods = len(batch[0][0])
        mods_stacked = []
        for i in range(num_mods):
            mods_stacked.append(torch.stack([m[i] for m in mod_lists]))
        labels = torch.tensor(labels)
        return mods_stacked, labels

    base_path = args.file_path  # note: use file_path, not data_dir

    train_dataset = PAMDataset(os.path.join(base_path, "train_all_subjects.pkl"), args.tt_max)
    if args.train_sample_ratio < 1.0:
        total = len(train_dataset)
        num = int(total * args.train_sample_ratio)
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(total, num, replace=False)
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        PAMDataset(os.path.join(base_path, "val_all_subjects.pkl"), args.tt_max),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        PAMDataset(os.path.join(base_path, "test_all_subjects.pkl"), args.tt_max),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    return train_loader, val_loader, test_loader

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
            BioBert, BioBertConfig, tokenizer = loadBert(args,device)
        else:
            BioBert, tokenizer = None, None

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
            train_dataloader, val_dataloader, test_data_loader = get_pam_dataloaders(args)

        
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
        if args.modeltype == 'Text':
            model = TextModel(args=args, device=device, orig_d_txt=768, Biobert=BioBert)
        elif args.modeltype == 'TS':
            model = TSMixed(args=args, device=device, orig_d_ts=30, orig_reg_d_ts=60, ts_seq_num=args.tt_max)
        else:
            model = MULTCrossModel(args=args, device=device, orig_d_ts=30, orig_reg_d_ts=60, orig_d_txt=768,
                                ts_seq_num=args.tt_max, text_seq_num=args.num_of_notes, Biobert=BioBert)
    elif args.dataset == "pam":
        # PAM modalities: chest(13), hand(13), ankle(13), heart_rate(1)
        modality_dims = [13, 13, 13, 1]
        # Ensure cross_method is set (from args)
        from core.model import FlexiModalMULTCrossModel
        model = FlexiModalMULTCrossModel(args, device, modality_dims)

    if args.modeltype=='TS':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
    elif args.modeltype=='TS_CXR':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
    elif 'Text' in args.modeltype:
        optimizer= torch.optim.Adam([
                {'params': [p for n, p in model.named_parameters() if 'bert' not in n]},
                {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': args.txt_learning_rate}
            ], lr=args.ts_learning_rate)
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
    print(f'Results saved in:\n{args.ck_file_path}')


if __name__ == "__main__":

    import time
    start_time = time.time()
    main()
    # wandb.finish()
    print("--- %s seconds ---" % (time.time() - start_time))
