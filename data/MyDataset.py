import numpy as np
import pandas as pd
import torch, torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
from dataload_collection import PATH_dict

#PATH = PATH_dict['']    # add 2017 to mother dict first
PATH = 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/BlobArchive/'


def _load_kernel_file(path):

    picklefile = open(pickle my ass path )
    return picklfile
    folders = os.listdir(path)
    data_list = list(str())
    for folder in folders:
        print(path + folder)
        with os.listdir(path + folder) as entries:
            for entry in entries:
                data_list.append(folder + entry.name.split(".")[0] + '.npy')
    return data_list

class Mask_n_pad(object):
    """
    Args:
        H, W: for the images
        (decided from the the biggest image under the 96% threshold)
    """
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image):
        """
        - Remove background with mask
        - Zeropad images up to the dimension of the biggest images - following the guide lines
        """
        img = 'hej'

        return img



T = transforms.Compose([Mask_n_pad(H=200,W=89),transforms.ToTensor(),transforms.Normalize(mean=[1,1,1,1,1,1,1],std=[1,1,1,1,1,1,1])])

class KornDataset(Dataset):

    def __init__(self, data_path,  label_path = None, transform = None):
        self.data_files = os.listdir(data_path)
        self.labels = pd.read_csv(label_path)
        self.transform = transform


    def __getitem__(self, index):
        img = np.load(self.data_files[index])

        label = None

        if self.labels:
            label =  self.labels.iloc[index,1]  # retrive the label coresponding to the image

        if self.transform:
            img = self.transform(img)



        return img, label



    def __len__(self):
        return len(self.data_files)


Dataset = KornDataset(data_path='...',  label_path = '...', transform = T)  # the dataset object can be indexed like a regular list
loader = DataLoader(Dataset, num_workers=8)
