import torch
import torch.nn as nn
import torch.nn.functional as F

class cvrlTrainer():
	def __init__(self, log_dir, model, train_loader, test_loader, optimizer, scheduler):
		self.log_dir = log_dir
		self.model = model
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.optimizer = optimizer
		self.scheduler = scheduler
