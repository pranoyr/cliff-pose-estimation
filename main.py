import os
import torch
import random
import os
import torch
import matplotlib.pyplot as plt
from cliff import CLIFF, get_config, Trainer

# add path for demo utils functions 
import numpy as np
import wandb
import argparse


def set_seed(seed):
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	os.environ['PYTHONHASHSEED'] = str(seed)
	

def main():
	set_seed(42)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.set_float32_matmul_precision('high')

	parser = argparse.ArgumentParser(description='CLIFF')
	parser.add_argument('--cfg', type=str, help='config file path')
	opt = parser.parse_args()

	# load config
	cfg = get_config(opt.cfg)
	# init wandb
	wandb.init(project=cfg.MODEL.NAME, config=cfg, name=cfg.EXP_NAME)

	# load model
	model = CLIFF(cfg).to(device)
	model = torch.compile(model)
	
	trainer = Trainer(cfg, model, device)
	trainer.fit()

	
if __name__ == '__main__':
	main()


