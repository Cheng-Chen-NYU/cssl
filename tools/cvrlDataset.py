import glob
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class cvrlDataset(Dataset):
	def __init__(self):
		super(cvrlDataset, self).__init__()