import numpy as np
import pandas as pd
import torch, torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
# from dataload_collection import PATH_dict
import pickle
import matplotlib.pyplot as plt
import timeit


# PATH = PATH_dict['']    # add 2017 to mother dict first
# PATH = 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/BlobArchive/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _load_pickle_file(path):
    infile = open(path, 'rb')
    pic = pickle.load(infile)
    infile.close()
    images = np.ones(len(pic), dtype=object)
    for i, image in enumerate(pic):
        images[i] = image['image']
    return images


def _make_data_list(root_path: str):
    folders = os.listdir(root_path)
    data_list = list(str())
    for folder in folders:
        with os.scandir(root_path + folder) as entries:
            for entry in entries:
                data_list.append(root_path + folder + '/' + entry.name.split(".")[0] + '.npy')
    return data_list[:10000]


class Mask_n_pad(object):
    """
    Args:
        H, W: for the images
        (decided from the the biggest image under the 92% threshold)
    """

    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, img):
        """
        - Crop image
        - Remove background with mask
        - Zeropad images up to the dimension of the biggest images - following the guide lines
        """

        # Apply mask

        mask = img[:, :, 7]
        img = np.where(mask[..., None] != 0, img, [0., 0., 0., 0., 0., 0., 0., 0.])

        # Trim/Crop image
        img = np.delete(img, np.where(np.sum(mask, axis=1) == 0)[0], axis=0)
        h = np.shape(img[:, :, 7])[0]
        img = np.delete(img, np.where(np.sum(mask, axis=0) == 0)[0], axis=1)
        w = np.shape(img[:, :, 0])[1]

        if (w > 80) or (h > 180):

            plt.imshow(img[:,:,4], img[:,:,2], img[:,:,1])
            plt.show()
            raise Exception('Image is too large. Larger than width:', self.W, 'or height', self.H)
        else:
            if (h % 2) == 0:
                rh1 = (self.H - h) / 2
                rh2 = (self.H - h) / 2
            elif (h % 2) == 1:
                rh1 = (self.H - h + 1) / 2
                rh2 = (self.H - h - 1) / 2
            if (w % 2) == 0:
                rw1 = (self.W - w) / 2
                rw2 = (self.W - w) / 2
            elif (w % 2) == 1:
                rw1 = (self.W - w + 1) / 2
                rw2 = (self.W - w - 1) / 2

            # Zero padding
            img = np.pad(img, ((int(rh2), int(rh1)), (int(rw1), int(rw2)), (0, 0)), 'constant')
            return img.astype('float32')



class KornDataset(Dataset):

    def __init__(self, data_path, label_path=None, transform=None):
        self.label_path = label_path
        self.data_files = _make_data_list(data_path)
        self.transform = transform
        self.get_label = False
        if self.label_path is not None:
            self.labels = pd.read_csv(label_path).set_index(['Names'])
            self.get_label = True

    def __getitem__(self, index):
        img = np.load(self.data_files[index])

        if self.transform:
            img = self.transform(img)


        if self.get_label:
            im = os.path.basename(os.path.normpath(self.data_files[index])).split(".")[0]
            label = str(self.labels.loc[im][0:7][self.labels.loc[im][0:7] == True]).split(' ')[0]
            return img, label

        return img, 'N/A'

    def __len__(self):
        return len(self.data_files)


if __name__ == '__main__':

    MEAN = np.load('../MEAN.npy')
    STD = np.load('../STD.npy')

    PATH_dict = {
        '10K_remote': 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenkblobs/',
        '10K_gamer': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/tenkblobs/',
        '224': 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/Foss_student/tenhblobsA/',
        'validation_grain': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/validation_grain/',
        'validation_blob': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/validation_blob/',
        'grainmix': 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/grainmix/'
    }
    DATA_SET = '10K_gamer'
    PATH = PATH_dict[DATA_SET]

    # T = transforms.Compose([Mask_n_pad(H=180, W=80), transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
    # start_time1 = timeit.default_timer()
    # traindata = KornDataset(data_path=PATH+'/train/', transform=T)
    # trainload = DataLoader(traindata, batch_size=3000, shuffle=True, num_workers=0, pin_memory=False)
    #
    # for batch_num, (X, _) in enumerate(trainload):
    #     # Regeneration and loss
    #     print(batch_num)
    # end_time1 = start_time = timeit.default_timer() - start_time1

    S = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
    start_time2 = timeit.default_timer()
    td = KornDataset(data_path=PATH, transform=S)
    tl = DataLoader(td, batch_size=3000, shuffle=True, num_workers=0, pin_memory=False)

    for batch_num, (X, _) in enumerate(tl):
        # Regeneration and loss
        print(batch_num)
    end_time2 = start_time = timeit.default_timer() - start_time2


    #print(end_time1)
    print(end_time2)







'''
# Test of classes
remove = Mask_n_pad(H=180,W=80)
img = np.load('C:/users/nullerh/desktop/temp/5c2f4c564162bb128cfb1600.npy')
post = remove(img)
fig, ar = plt.subplots(1,2)
ar[0].imshow(np.dstack((img[:,:,4], img[:,:,2], img[:,:,1])))
ar[1].imshow(np.dstack((post[:,:,4], post[:,:,2], post[:,:,1])))
fig.show()
print(np.shape(img), np.shape(post))
'''
