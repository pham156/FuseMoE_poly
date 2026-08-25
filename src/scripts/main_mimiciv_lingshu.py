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
from scripts.runtime_helpers import (
    get_pam_dataloaders as shared_get_pam_dataloaders,
    load_tokenizer_only as shared_load_tokenizer_only,
)


class Struct(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def load_tokenizer_only(args):
    return shared_load_tokenizer_only(args)

def get_pam_dataloaders(args):
    return shared_get_pam_dataloaders(args)

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

    run_name = build_run_name(args, prefix="fusemoe_new")

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
