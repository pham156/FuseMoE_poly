import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

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

def main():
    args = parse_args()

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

    wandb.init(
        project="polymoe_test",
        name=run_name,
        config=vars(args),
        dir="../../wandb"
    )


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

    if args.seed==0:
        copy_file(args.ck_file_path+'model/', src=os.getcwd())
    if args.mode=='train':
        if 'Text' in args.modeltype:
            BioBert, BioBertConfig, tokenizer = loadBert(args,device)
        else:
            BioBert, tokenizer = None, None

        from preprocessing.data_mimiciv import data_perpare

        train_dataset, train_sampler, train_dataloader = data_perpare(args, 'train', tokenizer)
        val_dataset, val_sampler, val_dataloader = data_perpare(args, 'val', tokenizer)
        _, _, test_data_loader = data_perpare(args,'test',tokenizer)

    if args.modeltype == 'Text':
        # pure text
        model= TextModel(args=args,device=device,orig_d_txt=768,Biobert=BioBert)
    elif args.modeltype == 'TS':
        # pure time series
        model= TSMixed(args=args,device=device,orig_d_ts=30,orig_reg_d_ts=60, ts_seq_num=args.tt_max)
    else:
        # multimodal fusion
        model= MULTCrossModel(args=args,device=device,orig_d_ts=30, orig_reg_d_ts=60, orig_d_txt=768,ts_seq_num=args.tt_max,text_seq_num=args.num_of_notes,Biobert=BioBert)
    print(device)
    
    if args.modeltype=='TS':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
    elif args.modeltype=='TS_CXR':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
    elif 'Text' in args.modeltype:
        optimizer= torch.optim.Adam([
                {'params': [p for n, p in model.named_parameters() if 'bert' not in n]},
                {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': args.txt_learning_rate}
            ], lr=args.ts_learning_rate)
    else:
        raise ValueError("Unknown modeltype in optimizer.")

    model, optimizer, train_dataloader,val_dataloader,test_data_loader = \
    accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, test_data_loader)

    best_model_state, best_val_score = trainer_irg(model=model,args=args,accelerator=accelerator,train_dataloader=train_dataloader,\
        dev_dataloader=val_dataloader, test_data_loader=test_data_loader, device=device,\
        optimizer=optimizer,writer=writer)
    # eval_test(args,model,test_data_loader, device)
    print(f"Best validation ({args.primary_metric}): {best_val_score}")
    model.load_state_dict(best_model_state)

    test_val = evaluate_irg(
        args=args,
        device=device,
        data_loader=test_data_loader,
        model=model,
        mode="test"
    )
    print("\n===== FINAL TEST RESULTS =====")
    for k, v in test_val.items():
        wandb.log({f"test/{k}": v})
        print(f"{k}: {v}")
    print("================================\n")

        

    print(f"New maximum memory allocated on GPU: {torch.cuda.max_memory_allocated(device)} bytes")
    print(f'Results saved in:\n{args.ck_file_path}')


if __name__ == "__main__":

    import time
    start_time = time.time()
    main()
    wandb.finish()
    print("--- %s seconds ---" % (time.time() - start_time))
