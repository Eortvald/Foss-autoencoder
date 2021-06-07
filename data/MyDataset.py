import numpy as np
import torch, torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
class MyDataset(torch.utils.Dataset):
    def __init__(self):
        self.data_files = data_list
        sorted(self.data_files)

    def __getindex__(self, idx):
        return np.load(self.data_files[idx])

    def __len__(self):
        return len(self.data_files)

path = 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/BlobArchive/'
folders = os.listdir(path)
data_list = list()
for folder in folders:
    with os.listdir(path + folder) as entries:
        for entry in entries:
            data_list.append(folder + entry.name.split(".")[0] + '.npy')

dset = MyDataset()
loader = torch.utils.DataLoader(dset, num_workers = 8)