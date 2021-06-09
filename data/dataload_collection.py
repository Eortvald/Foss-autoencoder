import numpy as np
import torch, torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import datetime
from os import listdir

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# PATHS
PATH_dict = {'10K': 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenkblobs/',
        'gamer': 'C:/Data/DeepEye/Foss_student/tenkblobs/'}

### If training on Foss Laptop select '10K'
### If training on Gamer select 'gamer'
PATH = PATH_dict['10K']

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

    folder = listdir(path)
    folder_images = len(folder)
    for i, NPY in enumerate(folder):

        if i % 10 == 0:

            print(f'Images loaded: [{i}/{folder_images}]  ------  {str(datetime.datetime.now())[11:-7]}')

        img, label = np.load(path+NPY, allow_pickle=True)
        data_x.append(img)

        numeric_label = make_numeric[label]
        data_y.append(numeric_label)

        #print(i,numeric_label)

    print(f'Done reading {subset} images')
    wx = torch.tensor(data_x, dtype=torch.float)
    tx = wx.permute(0, 3, 1, 2)
    ty = torch.tensor(data_y, dtype=torch.float)
    print(f'Dimension of x is :{tx.size()}')
    return tx, ty

xtrain, ytrain = npy_dir(PATH, 'train/')

xtrain, ytrain = xtrain.to(device), ytrain.to(device)

#train_data = TensorDataset(xtrain, ytrain)
Ktrain_loader = DataLoader(TensorDataset(xtrain, ytrain), batch_size=100, num_workers=0, shuffle=True)

xtest, ytest = npy_dir(PATH, 'test/')
xtest, ytest = xtest.to(device), ytest.to(device)
#test_data = TensorDataset(xtest, ytest)
Ktest_loader = DataLoader(TensorDataset(xtest, ytest), batch_size=100, num_workers=0, shuffle=True)

if __name__ == "__main__":

    pass

    #for b_index, (X, y) in enumerate(Ktest_loader):
     #   if b_index % 10 == 0:
      #      print(f'Batch number:{b_index} | Image:{X[0]}\n Image Label:{y[0]}')


