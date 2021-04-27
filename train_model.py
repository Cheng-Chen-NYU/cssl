import os
import shutil
import json
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tools.cvrlDataset import cvrlDataset
from tools.cvrlTrainer import cvrlTrainer
from model.model import MoCov1, MoCov2, SimCLRv1, SimCLRv2

parser = argparse.ArgumentParser(description='Contrastive Visual Representation Learning')
parser.add_argument("-name", "--name", type=str, help="name of the experiment", default="train_mocov1_on_cifar10")
parser.add_argument("-bs", "--batch_size", type=int, help="batch size", default=512)
parser.add_argument("-lr", type=float, help="learning rate", default=1e-3)
parser.add_argument("-momentum", type=float, help="momentum", default=0.9)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=1e-4)
parser.add_argument("-step_size", type=float, help="decay lr every step epochs", default=10)
parser.add_argument("-gamma", type=float, help="lr decay factor", default=0.5)
parser.add_argument("-arch", type=str, help="backbone cnn architecture name", default="resnet")

# data
parser.add_argument("-data_path", type=str, default="data/")

#
parser.add_argument("-results_dir", type=str, help="path to cache (default: none)", default='')
parser.add_argument("-resume", type=str, default='')

def create_folder(log_dir):
# make summary folder
	if not os.path.exists(log_dir):
		os.mkdir(log_dir)
	else:
		print('WARNING: summary folder already exists!! It will be overwritten!!')
		shutil.rmtree(log_dir)
		os.mkdir(log_dir)

if __name__ == '__main__':
	args = parser.parse_args()

	if args.results_dir == '':
		args.results_dir = '/cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-{}".format(args.name.split('_')[1]))

	log_dir = args.name + args.results_dir
	create_folder(log_dir)
	config_f = open(os.path.join(log_dir, 'config.json'), 'w')
	json.dump(vars(args), config_f)
	config_f.close()

	model_name = args.name.split('_')[1]
	if model_name == 'mocov1':
		model = MoCov1()
	elif model_name == 'mocov2':
		model = MoCov2()
	elif model_name == 'simclrv1': 
		model = SimCLRv1(arch='resnet18')
	elif model_name == 'simclrv2':
		model = SimCLRv2(arch='resnet18')
	else:
		assert(False)

	print('success')
	#