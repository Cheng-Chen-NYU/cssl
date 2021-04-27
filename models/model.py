import torch
import torch.nn as nn
from torchvision.models import resnet

from tqdm import tqdm

class MoCov1(nn.Module):
	def __init__(self, feature_dim=512, K=4096, m=0.99, T=0.1, arch='resnet18', bn_splits=8):
		super(MoCov1, self).__init__()

		self.feature_dim = feature_dim
		self.K = K
		self.m = m
		self.T = T

	def forward(self, x1, x2):
		pass

class MoCov2(nn.Module):
	def __init__(self, dim=512, K=4096, m=0.99, T=0.1, arch='resnet18', bn_splits=8):
		super(MoCov2, self).__init__()

	def forward(self, x):
		pass

class SimCLRv1(nn.Module):
	def __init__(self, feature_dim=128, arch='resnet50'):
		super(SimCLRv1, self).__init__()

		self.f = []
		for name, module in getattr(resnet, arch)().named_children():
			if name == 'conv1':
				module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
			if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
				self.f.append(module)
		# encoder
		self.f = nn.Sequential(*self.f)
		# projection head
		self.g = nn.Sequential(
							nn.Linear(2048, 512, bias=False),
							nn.BatchNorm1d(512),
							nn.ReLU(inplace=True), 
							nn.Linear(512, feature_dim, bias=True)
						)

	def forward(self, x):
		x = self.f(x)
		feature = torch.flatten(x, start_dim=1)
		out = self.g(feature)
		return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

class SimCLRv2(nn.Module):
	def __init__(self, feature_dim=128, arch='resnet50'):
		super(SimCLRv1, self).__init__()

	def forward(self, x):
		pass

