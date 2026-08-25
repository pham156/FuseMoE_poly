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
import copy
logger = logging.getLogger(__name__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import *
from core.train import *
from utils.checkpoint import *
from utils.util import *
from accelerate import Accelerator
from core.interp import *
from scripts.runtime_helpers import get_pam_dataloaders as shared_get_pam_dataloaders


class Struct(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)

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


def _build_text_backbone(args, device):
    if 'Text' not in args.modeltype:
        return None, None
    BioBert, _, tokenizer = loadBert(args, device)
    return BioBert, tokenizer


def _build_model(args, device, BioBert=None, tokenizer=None, pam_modality_dims=None):
    if args.dataset == "mimic":
        if args.modeltype == 'Text':
            return TextModel(args=args, device=device, orig_d_txt=768, Biobert=BioBert)
        if args.modeltype == 'TS':
            return TSMixed(args=args, device=device, orig_d_ts=30, orig_reg_d_ts=60, ts_seq_num=args.tt_max)
        return MULTCrossModel(
            args=args,
            device=device,
            orig_d_ts=30,
            orig_reg_d_ts=60,
            orig_d_txt=768,
            ts_seq_num=args.tt_max,
            text_seq_num=args.num_of_notes,
            Biobert=BioBert,
            tokenizer=tokenizer,
        )

    from core.model import FlexiModalMULTCrossModel
    model = FlexiModalMULTCrossModel(args, device, pam_modality_dims)
    return model


def _save_best_checkpoint(args, model_state, best_val_score):
    if getattr(args, "disable_run_folder_save", False) or model_state is None:
        return None
    payload = {
        "network": model_state,
        "best_val_score": best_val_score,
        "seed": args.seed,
        "task": args.task,
        "modeltype": args.modeltype,
    }
    checkpoint_path = None
    if args.ck_file_path is not None:
        os.makedirs(args.ck_file_path, exist_ok=True)
        checkpoint_path = os.path.join(args.ck_file_path, "best_model.pth.tar")
        torch.save(payload, checkpoint_path)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(payload, os.path.join(args.output_dir, "best_model.pth.tar"))
    return checkpoint_path


def _task_short_name(task_name):
    if "pheno" in task_name:
        return "pheno"
    if "los" in task_name:
        return "los"
    return "ihm"


def _task_num_labels(task_name):
    return 25 if _task_short_name(task_name) == "pheno" else 2


def _task_primary_metric(task_name):
    return "macro_f1" if _task_short_name(task_name) == "pheno" else "f1"


class RoundRobinLoader:
    def __init__(self, loaders):
        self.loaders = loaders
        self.task_names = list(loaders.keys())
        self._length = sum(len(loader) for loader in loaders.values())

    def __len__(self):
        return self._length

    def __iter__(self):
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        active = list(self.task_names)
        while active:
            next_active = []
            for task_name in active:
                try:
                    batch = next(iterators[task_name])
                except StopIteration:
                    continue
                if batch is None:
                    next_active.append(task_name)
                    continue
                batch = list(batch)
                metadata = batch[19] if len(batch) > 19 else {}
                metadata = dict(metadata)
                metadata["task_name"] = _task_short_name(task_name)
                batch[19] = metadata
                yield tuple(batch)
                next_active.append(task_name)
            active = next_active


def _build_multitask_mimic_loaders(args, tokenizer):
    from preprocessing.data_mimiciv import data_perpare

    task_specs = [task.strip() for task in str(args.multitask_tasks).split(",") if task.strip()]
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}
    for task_name in task_specs:
        task_args = copy.deepcopy(args)
        task_args.task = task_name
        task_args.num_labels = _task_num_labels(task_name)
        task_args.primary_metric = _task_primary_metric(task_name)
        _, _, train_loaders[_task_short_name(task_name)] = data_perpare(task_args, 'train', tokenizer)
        _, _, val_loaders[_task_short_name(task_name)] = data_perpare(task_args, 'val', tokenizer)
        _, _, test_loaders[_task_short_name(task_name)] = data_perpare(task_args, 'test', tokenizer)
    return RoundRobinLoader(train_loaders), val_loaders, test_loaders

def main():
    args = parse_args()
    prepare_expert_init_targets(args)
    prepare_unimodal_teacher_targets(args)

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
    BioBert, tokenizer = _build_text_backbone(args, device)
    pam_class_weights = None
    pam_modality_dims = None

    if args.use_instruction_router:
        if BioBert is None:
            raise ValueError("--use_instruction_router requires a modeltype that includes Text.")
        args.router_instruction_dim = BioBert.config.hidden_size
    else:
        args.router_instruction_dim = None

    if args.mode == 'train':
        if args.dataset == "mimic":
            from preprocessing.data_mimiciv import data_perpare, TSNote_Irg, TextTSIrgcollate_fn

            if args.multitask_shared_moe_trunk:
                train_dataloader, val_dataloader, test_data_loader = _build_multitask_mimic_loaders(args, tokenizer)
            else:
                train_dataset = TSNote_Irg(args, 'train', tokenizer)
                _, _, val_dataloader = data_perpare(args, 'val', tokenizer)
                _, _, test_data_loader = data_perpare(args, 'test', tokenizer)

                if args.train_sample_ratio < 1.0:
                    total = len(train_dataset)
                    num_samples = int(total * args.train_sample_ratio)
                    rng = np.random.RandomState(args.seed)

                    if args.task in ['ihm', 'los']:
                        labels = [train_dataset[i]['label'].item() for i in range(total)]
                        perm = get_stratified_permutation(train_dataset, labels, rng)
                    else:
                        perm = rng.permutation(total)

                    indices = perm[:num_samples]
                    train_subset = Subset(train_dataset, indices)
                    train_dataloader = DataLoader(
                        train_subset,
                        batch_size=args.train_batch_size,
                        shuffle=True,
                        collate_fn=TextTSIrgcollate_fn,
                        num_workers=0,
                    )
                    print(f"Training with {len(train_subset)} samples (ratio={args.train_sample_ratio})")
                else:
                    _, _, train_dataloader = data_perpare(args, 'train', tokenizer)
        else:
            train_dataloader, val_dataloader, test_data_loader, pam_class_weights, pam_modality_dims = get_pam_dataloaders(args)

        model = _build_model(args, device, BioBert=BioBert, tokenizer=tokenizer, pam_modality_dims=pam_modality_dims)
        if args.dataset == "pam":
            weight_tensor = torch.tensor(pam_class_weights, dtype=torch.float32, device=device)
            model.loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)

        if args.modeltype in ['TS', 'TS_MOE', 'TS_CXR', 'pam']:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
        elif 'Text' in args.modeltype:
            optimizer = torch.optim.Adam([
                {'params': [p for n, p in model.named_parameters() if 'bert' not in n]},
                {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': args.txt_learning_rate}
            ], lr=args.ts_learning_rate)
        else:
            raise ValueError("Unknown modeltype in optimizer.")

        if args.multitask_shared_moe_trunk and args.dataset == "mimic":
            model, optimizer = accelerator.prepare(model, optimizer)
        else:
            model, optimizer, train_dataloader, val_dataloader, test_data_loader = accelerator.prepare(
                model, optimizer, train_dataloader, val_dataloader, test_data_loader
            )

        best_model_state, best_val_score = trainer_irg(
            model=model,
            args=args,
            accelerator=accelerator,
            train_dataloader=train_dataloader,
            dev_dataloader=val_dataloader,
            test_data_loader=test_data_loader,
            device=device,
            optimizer=optimizer,
            writer=writer,
            dataset=args.dataset,
        )
        print(f"Best validation ({args.primary_metric}): {best_val_score}")
        if best_model_state is None:
            raise RuntimeError("Training completed without a best model state.")
        accelerator.unwrap_model(model).load_state_dict(best_model_state)
        checkpoint_path = _save_best_checkpoint(args, best_model_state, best_val_score)
        if checkpoint_path is not None:
            print(f"Saved best checkpoint to: {checkpoint_path}")

        if getattr(args, "use_r2t2_rerouting", False):
            if args.multitask_shared_moe_trunk and isinstance(val_dataloader, dict):
                raise RuntimeError("R2T2 rerouting is currently implemented for single-task runs only.")
            print("\n===== BUILDING R2T2 VALIDATION REFERENCE BANK =====")
            build_r2t2_adaptor(
                args=args,
                device=device,
                val_dataloader=val_dataloader,
                model=model,
                dataset=args.dataset,
            )
            print("===============================================\n")

        print("\n===== FINAL TEST RESULTS =====")
        if args.multitask_shared_moe_trunk and isinstance(test_data_loader, dict):
            for task_name, task_loader in test_data_loader.items():
                task_args = copy.deepcopy(args)
                task_args.task = task_name
                task_args.num_labels = _task_num_labels(task_name)
                test_val = evaluate_irg(
                    args=task_args,
                    device=device,
                    data_loader=task_loader,
                    model=model,
                    mode="test",
                    dataset=args.dataset
                )
                print(f"[{task_name}]")
                for k, v in test_val.items():
                    print(f"{k}: {v}")
        else:
            test_val = evaluate_irg(
                args=args,
                device=device,
                data_loader=test_data_loader,
                model=model,
                mode="test",
                dataset=args.dataset
            )
            for k, v in test_val.items():
                print(f"{k}: {v}")
        print("================================\n")
    elif args.mode == "eval":
        if getattr(args, "use_r2t2_rerouting", False):
            raise RuntimeError(
                "R2T2 eval-only replay is not implemented. Use --use_r2t2_rerouting during training "
                "so the validation bank and test rerouting run from the selected in-memory model."
            )
        if not args.checkpoint_path:
            raise ValueError("--checkpoint_path is required in --mode eval.")
        if args.dataset == "mimic":
            from preprocessing.data_mimiciv import data_perpare
            _, _, eval_dataloader = data_perpare(args, args.eval_split, tokenizer)
        else:
            train_dataloader, val_dataloader, test_data_loader, pam_class_weights, pam_modality_dims = get_pam_dataloaders(args)
            if args.eval_split == "train":
                eval_dataloader = train_dataloader
            elif args.eval_split == "val":
                eval_dataloader = val_dataloader
            else:
                eval_dataloader = test_data_loader

        model = _build_model(args, device, BioBert=BioBert, tokenizer=tokenizer, pam_modality_dims=pam_modality_dims)
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        state_dict = checkpoint.get("network", checkpoint)
        model.load_state_dict(state_dict)
        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
        eval_vals = evaluate_irg(
            args=args,
            device=device,
            data_loader=eval_dataloader,
            model=model,
            mode="test" if args.eval_split == "test" else "val",
            dataset=args.dataset,
        )
        print(f"\n===== EVAL RESULTS ({args.eval_split}) =====")
        for k, v in eval_vals.items():
            print(f"{k}: {v}")
        print("=================================\n")
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

        

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
