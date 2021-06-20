import os
import numpy as np
import datetime

def mask_n_pad_all(root_path, max_h = 180, max_w = 180):
    folders = ['train/', 'test/']

    # Going through both train and test set
    for folder in folders:

        # looping through sub folders
        sub_folders = os.listdir(root_path + folder)
        for sub in sub_folders:

            # Saving temporary path
            temp_path = root_path + folder + sub

            # looping through each image
            with os.scandir(temp_path) as entries:
                for entry in entries:
                    img = np.load(temp_path + entry)
                    print(img.dtype, 'before process')

                    #Apply mask
                    mask = img[:, :, 7]
                    img = np.where(mask[..., None] != 0, img, [0, 0, 0, 0, 0, 0, 0, 0])

                    # Trim/Crop image
                    img = np.delete(img, np.where(np.sum(mask, axis=1) == 0)[0], axis=0)
                    h = np.shape(img[:, :, 7])[0]
                    img = np.delete(img, np.where(np.sum(mask, axis=0) == 0)[0], axis=1)
                    w = np.shape(img[:, :, 0])[1]

                    if (w > 80) or (h > 180):
                        np.delete(temp_path + entry)
                        print('image', entry, 'was too large')
                    else:
                        if (h % 2) == 0:
                            rh1 = (max_h - h) / 2
                            rh2 = (max_h - h) / 2
                        elif (h % 2) == 1:
                            rh1 = (max_h - h + 1) / 2
                            rh2 = (max_h - h - 1) / 2
                        if (w % 2) == 0:
                            rw1 = (max_w - w) / 2
                            rw2 = (max_w - w) / 2
                        elif (w % 2) == 1:
                            rw1 = (max_w - w + 1) / 2
                            rw2 = (max_w - w - 1) / 2

                        # Zero padding
                        img = np.pad(img, ((int(rh2), int(rh1)), (int(rw1), int(rw2)), (0, 0)), 'constant')

                        # Converting to float
                        img = img.astype('float')
                        print(img, 'after process')

                        # Saving image
                        np.save(temp_path + entry, img)


root_paths = ['C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/validation_grain/',
              'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/validation_blob/']
for root_path in root_paths:
    mask_n_pad_all(root_path)