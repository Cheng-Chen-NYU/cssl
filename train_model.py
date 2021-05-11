import os
import json
import argparse
import numpy as np
from datetime import datetime
from thop import profile, clever_format

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from tools.cvrlDataset import CIFAR10Pair, train_transform, test_transform
from tools.cvrlTrainer import mocoTrainer, simclrTrainer
from model.model import MoCov1, MoCov2, SimCLRv1, SimCLRv2

parser = argparse.ArgumentParser(description='Contrastive Visual Representation Learning')

# model
parser.add_argument('--model_name', type=str, help='model name', default='mocov1')
parser.add_argument('--epochs', type=int, help='number of epochs', default=200)

# data loader
parser.add_argument('--batch_size', type=int, help='batch size', default=512)

# optimizer
parser.add_argument('--learning_rate', type=float, help='learning rate', default=1e-3)
parser.add_argument('--momentum', type=float, help='optimizer momentum', default=0.9)
parser.add_argument('--weight_decay', type=float, help='weight decay factor', default=1e-6)

# scheduler
parser.add_argument('--step_size', type=float, help='decay lr every step epochs', default=10)
parser.add_argument('--gamma', type=float, help='lr decay factor', default=0.5)
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

# backbone
parser.add_argument('--arch', type=str, help='backbone cnn architecture', default='resnet18')

# softmax
parser.add_argument('--temperature', default=0.5, type=float, help='temperature used in softmax')

# moco
parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
parser.add_argument('--moco-k', default=4096, type=int, help='queue size; number of negative keys')
parser.add_argument('--moco-m', default=0.99, type=float, help='moco momentum of updating key encoder')

# knn
parser.add_argument('--k', default=200, type=int, help='top k most similar images used to predict the label')

# path
parser.add_argument('--dataset_dir', type=str, default="data/")
parser.add_argument('--log_dir', type=str, help="dir to log", default='train_log/')
parser.add_argument('--results_dir', type=str, help='dir to cache (default: none)', default='')
parser.add_argument('--resume_path', type=str, help='path to latest checkpoint (default: none)', default='')

if __name__ == '__main__':

	args = parser.parse_args()

	if args.results_dir == '':
		args.results_dir = '/cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-{}".format(args.model_name))

	if not os.path.exists(args.log_dir):
		os.mkdir(args.log_dir)

	train_data = CIFAR10Pair(root=args.dataset_dir, train=True, transform=train_transform, download=True)
	train_iter = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

	memory_data = CIFAR10(root=args.dataset_dir, train=True, transform=test_transform, download=True)
	memory_iter = DataLoader(memory_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

	test_data = CIFAR10(root=args.dataset_dir, train=False, transform=test_transform, download=True)
	test_iter = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

	train_log = args.log_dir + args.results_dir

	model_name = args.model_name
	if model_name == 'mocov1':
	
		args.cos = True
		model = MoCov1(feature_dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.temperature, arch=args.arch, bn_splits=8).cuda()
		
		optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
		trainer = mocoTrainer(train_log, model, train_iter, memory_iter, test_iter, optimizer, args.temperature, args.k, args.learning_rate, args.cos)
	
	elif model_name == 'mocov2':
	
		args.cos = True
		model = MoCov2(feature_dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.temperature, arch=args.arch, bn_splits=8).cuda()
		
		optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
		trainer = mocoTrainer(train_log, model, train_iter, memory_iter, test_iter, optimizer, args.temperature, args.k, args.learning_rate, args.cos)
	
	elif model_name == 'simclrv1':
	
		model = SimCLRv1(arch=args.arch).cuda()
		
		# 24.62M 1.31G
		flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
		flops, params = clever_format([flops, params])
		print('# Model Params: {} FLOPs: {}'.format(params, flops))
		
		optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
		trainer = simclrTrainer(train_log, model, train_iter, memory_iter, test_iter, optimizer, args.temperature, args.k)
	
	elif model_name == 'simclrv2':

		model = SimCLRv2(arch=args.arch).cuda()
		
		# 26.19M 1.31G
		flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
		flops, params = clever_format([flops, params])
		print('# Model Params: {} FLOPs: {}'.format(params, flops))
		
		optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
		trainer = simclrTrainer(train_log, model, train_iter, memory_iter, test_iter, optimizer, args.temperature, args.k)
	
	else:
		assert(False)

	if not os.path.exists(train_log):
		os.mkdir(train_log)

	config_f = open(os.path.join(train_log, 'config.json'), 'w')
	json.dump(vars(args), config_f)
	config_f.close()

	epoch_start = 1
	trainer.train(args.resume_path, epoch_start, args.epochs)

