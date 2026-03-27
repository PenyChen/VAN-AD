import argparse
import time
import os

import yaml
from solver.vanad_sovler import VANAD_sovler

import logging
from logging.handlers import RotatingFileHandler
import random
import numpy as np
import torch
import torch.distributed as dist

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            filename='zz_log.log',
            maxBytes=1024 * 1024,
            backupCount=5,
            encoding='utf-8'
        )
    ],
    encoding='utf-8'
)
logger = logging.getLogger(__name__)
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank
    else:
        return False, 0
def setup_config(args, config, logger):
    """
    Display the configuration parameters for the experiment。
    """
    for attr in vars(args):
        if getattr(args, attr) is None:
            if attr in config:
                setattr(args, attr, config[attr])
            else:
                pass

    logger.info("┌" + "─" * 60 + "┐")
    logger.info(f"│ {'PARAMETER':<25} | {'VALUE':<30} │")
    logger.info("├" + "─" * 60 + "┤")

    for attr, value in vars(args).items():
        display_val = "N/A" if value is None else str(value)
        if len(display_val) > 30:
            display_val = display_val[:27] + "..."
        logger.info(f"│ {attr:<25} | {display_val:<30} │")
    logger.info("└" + "─" * 60 + "┘")

    # Experiment Settings
    important_fields = ["data", "mode", "epochs", "data_ratio", "arch", 
                        "finetune_type", "seq_len", "periodicity", "image_method", "image_mask"]
    prefix_list = []
    
    logger.info("-" * 40)
    logger.info("Experiment Settings Summary")
    logger.info("-" * 40)
    
    for k in important_fields:
        val = getattr(args, k, "N/A")
        logger.info(f"{k:<15}: {val}")
        
        if k != "mode":
            prefix_val = "{epochs}" if k == "epochs" else str(val)
            prefix_list.append(prefix_val)

    setattr(args, "logger", logger)
    setattr(args, "prefix", "_".join(prefix_list))
    logger.info(f"Generated prefix: {args.prefix}")
    
    return args

def main():
    print(f"DEBUG: Process started. RANK: {os.environ.get('RANK')}, LOCAL_RANK: {os.environ.get('LOCAL_RANK')}", flush=True)
    use_ddp, local_rank = setup_distributed()
    seed=42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    arg = argparse.ArgumentParser()
    arg.add_argument("--config", type=str, help='config file path')
    # model config
    arg.add_argument("--arch", type=str, help='Model architecture.')
    arg.add_argument("--finetune_type", type=str, help='Finetune type.')
    arg.add_argument("--ckpt_dir", type=str, help='Checkpoint directory.')
    arg.add_argument("--load_ckpt", type=bool, help='Load checkpoint.')
    arg.add_argument("--pred_len", type=int,  help='Prediction length.')
    arg.add_argument("--seq_len", type=int,  help='Sequence length.')
    arg.add_argument("--task", type=str, help="Task: 'rec' or 'forecast'")
    arg.add_argument("--periodicity", type=int,  help='Periodicity.')
    arg.add_argument("--image_method", type=str,  help="Image method: seg,gaf,stft,rp,wavl")
    arg.add_argument("--image_mask", type=str,  help="Image mask method: random or complementary")
    arg.add_argument("--norm_const", type=float,  help='Normalization constant.')
    arg.add_argument("--denormalize", type=bool, help='Denormalize.')
    arg.add_argument("--use_new_nf", type=bool, help='Use new normalizing flow implementation.')
    # nf config
    arg.add_argument("--use_nf", type=bool, help='Use normalizing flow.')
    arg.add_argument("--n_blocks", type=int,  help='Number of blocks.')
    arg.add_argument("--n_sensor", type=int,  help='Number of sensors.')
    arg.add_argument("--input_size", type=int,  help='')
    arg.add_argument("--hidden_size", type=int,  help='')
    arg.add_argument("--n_hidden", type=int,  help='')
    arg.add_argument("--cond_label_size", type=int,  help='')
    arg.add_argument("--activation", type=str,  help='')
    arg.add_argument("--input_order", type=str,  help='Input order.')
    arg.add_argument("--batch_norm", type=bool,  help='Batch normalization.')
    arg.add_argument("--lambda_nf", type=float,  help='Lambda of loss_nf')
    arg.add_argument("--lambda_rec", type=float,  help='Lambda of loss_rec')
    arg.add_argument("--nf_mode", type=str,  help='')
    # data config
    arg.add_argument("--data", type=str,  help='Dataset name.')
    arg.add_argument("--data_path", type=str,  help='Dataset path.')
    arg.add_argument("--batch_size", type=int,  help='Batch size.')
    arg.add_argument("--num_workers", type=int,  help='Number of workers.')
    arg.add_argument("--data_ratio", type=float,  help='Ratio of training data to use.')
    arg.add_argument("--num_vars", type=int,  help="Number of variables.")
    # use tab dataset
    arg.add_argument("--tab_csv_file", type=str)
    # exp config
    arg.add_argument("--mode", type=str, choices=["finetune", "finetune-test", "zero-test"], help='Mode: train or test.')
    arg.add_argument("--exp_name", type=str, help='Experiment name.')
    # arg.add_argument("--gpu", type=int,  help='GPU id.')
    arg.add_argument("--anormly_ratio", type=str,  help='Checkpoint path.')
    arg.add_argument("--pic_save_path", type=str,  help='Path to save picture.')
    arg.add_argument("--epochs", type=int,  help='Epochs.')
    arg.add_argument("--lr", type=float,  help='Learning rate.')

    args = arg.parse_args()
    
    if hasattr(args, "config"):
        logger.info(f"Loading config file from {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    args = setup_config(args, config, logger)
    
    args.use_ddp = use_ddp
    args.local_rank = local_rank
    solver = VANAD_sovler(args)
    logger.info("[Starting]")
    if args.mode == "zero-test":
        solver.test()
    elif args.mode == "nf-train":
        solver.nf_train() 
        solver.nf_test()
    elif args.mode == "nf-test":
        solver.nf_test()
    else:
        raise ValueError("Invalid mode")
    logger.info("[Finished]")
    return 
    
if __name__ == '__main__':
    main()