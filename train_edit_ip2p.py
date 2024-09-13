from code.gaussian_training import  Trainer
import torch
import argparse
import os
# from code.facetracking.shape2lms import get_ldmop
from datetime import datetime
import ipdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default='0_1',
                            help='processed source video directory')
    parser.add_argument("--prompt", type=str, default=None,
                        help=None)
    parser.add_argument("--train_skip", type=int, default=0)
    parser.add_argument("--exp_name",type=str,default="")
    args = parser.parse_args()
    source_dir = args.source_dir
    prompt = args.prompt
    train_skip = args.train_skip
    device = 'cuda:0'
    sid = source_dir.split('/')[-2]
    hid = prompt.replace(' ', "_")
    save_dir = f"output/{sid}/{hid}"
    os.makedirs(save_dir, exist_ok=True)
    diface_trainer = Trainer(source_dir, save_dir, device=device, dtype = torch.float32, prompt=prompt, train_skip=train_skip,exp_name=args.exp_name)
    diface_trainer.train_ip2p_fea()