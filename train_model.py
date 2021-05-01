import os
import json
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from tools.cvrlDataset import CIFAR10Pair, train_transform, test_transform
from tools.cvrlTrainer import cvrlTrainer
from model.model import MoCov1, MoCov2, SimCLRv1, SimCLRv2

parser = argparse.ArgumentParser(description='Contrastive Visual Representation Learning')
parser.add_argument("-name", "--name", type=str, help="name of the experiment", default="train_log")
parser.add_argument("-bs", "--batch_size", type=int, help="batch size", default=512)
parser.add_argument("-t", "--temperature", default=0.5, type=float, help='Temperature used in softmax')
parser.add_argument("-lr", "--learning_rate", type=float, help="learning rate", default=1e-3)
parser.add_argument("-m", "--momentum", type=float, help="momentum", default=0.9)
parser.add_argument("-wd", "--weight_decay", type=float, help="weight decay", default=1e-6)
parser.add_argument("-step_size", type=float, help="decay lr every step epochs", default=10)
parser.add_argument("-gamma", type=float, help="lr decay factor", default=0.5)
parser.add_argument("-arch", type=str, help="backbone cnn arch name", default="resnet18")
parser.add_argument("-model", type=str, help="model name", default="mocov1")

# KNN
parser.add_argument('-k', default=200, type=int, help='Top k most similar images used to predict the label')

# data
parser.add_argument("-data_path", type=str, default="data/")

#
parser.add_argument("-results_dir", type=str, help="path to cache (default: none)", default='')
parser.add_argument("-resume", type=str, default='')

if __name__ == '__main__':
	args = parser.parse_args()

	if args.results_dir == '':
		args.results_dir = '/cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-{}".format(args.arch))

	if not os.path.exists(args.name):
		os.mkdir(args.name)

	model_name = args.model
	if model_name == 'mocov1':
		model = MoCov1()
	elif model_name == 'mocov2':
		model = MoCov2()
	elif model_name == 'simclrv1': 
		model = SimCLRv1(arch=args.arch)
	elif model_name == 'simclrv2':
		model = SimCLRv2(arch=args.arch)
	else:
		assert(False)

	train_data = CIFAR10Pair(root=args.data_path, train=True, transform=train_transform, download=True)
	train_iter = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

	memory_data = CIFAR10(root=args.data_path, train=True, transform=test_transform, download=True)
	memory_iter = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

	test_data = CIFAR10(root=args.data_path, train=False, transform=test_transform, download=True)
	test_iter = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

	log_dir = args.name + args.results_dir
	if not os.path.exists(log_dir):
		os.mkdir(log_dir)

	config_f = open(os.path.join(log_dir, 'config.json'), 'w')
	json.dump(vars(args), config_f)
	config_f.close()

	epoch_start = 1
	c = len(memory_data.classes)
	trainer = cvrlTrainer(log_dir, model, train_iter, memory_iter, test_iter, optimizer, args.temperature, args.k)
	trainer.train(args.resume, c, epoch_start, 200)

