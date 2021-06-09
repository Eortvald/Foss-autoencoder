import numpy as np
import torch, torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from os import listdir

from data.dataload_collection import npy_dir, PATH


xtrain, ytrain = npy_dir(PATH['gamer'], 'train/')
#xtrain = torch.normal(mean=10, std=2, size=(100, 8, 10, 10))
#ytrain = torch.normal(mean=2, std=4, size=(100, 8, 10, 10))


dataset = TensorDataset(xtrain, ytrain)
statload = DataLoader(dataset, batch_size=len(dataset), num_workers=0, shuffle=False)

data = next(iter(statload))
mean = torch.mean(data[0], dim=(0, 2, 3)).numpy()
std = torch.std(data[0], dim=(0, 2, 3)).numpy()

np.save('10K_mean', mean)
np.save('10K_std', std)


