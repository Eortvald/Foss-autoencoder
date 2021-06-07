import numpy as np
import torch, torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

path = 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/BlobArchive/'

def _load_kernel_file(path):
    folders = os.listdir(path)
    data_list = list()
    for folder in folders:
        with os.listdir(path + folder) as entries:
            for entry in entries:
                data_list.append(folder + entry.name.split(".")[0] + '.npy')
    return data_list

class MyDataset(Dataset):
    def __init__(self, data_files):
        self.data_files = data_list
        sorted(self.data_files)

    def __getindex__(self, index):
        return np.load(self.data_files[index])

    def __len__(self):
        return len(self.data_files)


train_set = MyDataset()
loader = torch.utils.DataLoader(dset, num_workers = 8)