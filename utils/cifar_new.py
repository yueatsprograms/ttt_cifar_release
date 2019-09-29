import torch
import torch.utils.data as data
import numpy as np
from PIL import Image

class CIFAR_New(data.Dataset):
	def __init__(self, root, transform=None, target_transform=None, version='v6'):
		self.data = np.load('%s/cifar10.1_%s_data.npy' %(root, version))
		self.targets = np.load('%s/cifar10.1_%s_labels.npy' %(root, version)).astype('long')
		self.transform = transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]
		img = Image.fromarray(img)
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return img, target

	def __len__(self):
		return len(self.targets)