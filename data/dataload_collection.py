import numpy as np
import torch, torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from os import listdir

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# PATHS
PATH = {'10K': 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenkblobs/',
        'gamer': 'C:/Data/DeepEye/Foss_student/tenkblobs/'}


# Transforms


# Folder loading

def npy_dir(path: str, subset: str):
    path = path + subset

    make_numeric = {'Oat': 1,
                'Broken': 2,
                'Rye': 3,
                'Wheat': 4,
                'BarleyGreen': 5,
                'Cleaved': 6,
                'Skinned': 7}

    data_x = []
    data_y = []
    for i, NPY in enumerate(listdir(path)):

        img, label = np.load(path+NPY, allow_pickle=True)
        data_x.append(img)

        numeric_label = make_numeric[label]
        data_y.append(numeric_label)

        #print(i,numeric_label)

    print(f'Done reading {subset} images')
    wx = torch.tensor(data_x, dtype=torch.float)
    tx = wx.permute(0, 3, 1, 2)
    ty = torch.tensor(data_y, dtype=torch.float)

    return tx, ty


xtrain, ytrain = npy_dir(PATH['gamer'], 'train/')

xtrain, ytrain = xtrain.to(device), ytrain.to(device)

#train_data = TensorDataset(xtrain, ytrain)
Ktrain_loader = DataLoader(TensorDataset(xtrain, ytrain), batch_size=100, num_workers=0, shuffle=True)

xtest, ytest = npy_dir(PATH['gamer'], 'test/')
xtest, ytest = xtest.to(device), ytest.to(device)
#test_data = TensorDataset(xtest, ytest)
Ktest_loader = DataLoader(TensorDataset(xtest, ytest), batch_size=100, num_workers=0, shuffle=True)

if __name__ == "__main__":

    pass

    #for b_index, (X, y) in enumerate(Ktest_loader):
     #   if b_index % 10 == 0:
      #      print(f'Batch number:{b_index} | Image:{X[0]}\n Image Label:{y[0]}')


"""
def dataset_from_collection(PATH: str, batch_size: int, transform: object = None, filetype: str = '.npy'):
    npy_load = lambda PATH: torch.from_numpy(np.load(PATH, allow_pickle=True))

    train_data = datasets.DatasetFolder(root=PATH + '/train', loader=npy_load, extensions=([filetype]),
                                        transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2)

    test_data = datasets.DatasetFolder(root=PATH + '/test', loader=npy_load, extensions=([filetype]),
                                       transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2)

    return train_loader, test_loader


train_loader, test_loader = dataset_from_collection(PATH=PATH_dict['10K'], transform=Norm_transform, batch_size=128)
"""

