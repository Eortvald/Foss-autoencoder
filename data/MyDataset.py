import numpy as np
import torch, torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils import Dataset
import os

path = 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/BlobArchive/'

def _load_kernel_file(path):
    folders = os.listdir(path)
    data_list = list(str())
    for folder in folders:
        with os.listdir(path + folder) as entries:
            for entry in entries:
                data_list.append(folder + entry.name.split(".")[0] + '.npy')
    return np.load(data_list)[0]

class MyDataset(torch.utils.Dataset):
    def __init__(self, data_files):
        self.data_files = np.sort(data_files)

    def __getindex__(self, index):
        return _load_kernel_file(self.data_files[index])

    def __len__(self):
        return len(self.data_files)


train_files = np.load(path + '00/' + '5c2f4c564162bb128cfb1a00.npy')
train_set = MyDataset(data_files = train_files)
loader = torch.utils.DataLoader(train_set, num_workers = 8)