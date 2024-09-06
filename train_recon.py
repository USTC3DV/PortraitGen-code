from code.gaussian_training import Trainer
import torch
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default='0_1',
                            help='processed source video directory')
    parser.add_argument("--exp_name", type=str, default='')
    args = parser.parse_args()
    source_dir = args.source_dir
    device = 'cuda:0'

    save_dir = source_dir
    
    diface_trainer = Trainer(source_dir, save_dir, device=device, dtype = torch.float32)
    diface_trainer.train_gaussian_fea()