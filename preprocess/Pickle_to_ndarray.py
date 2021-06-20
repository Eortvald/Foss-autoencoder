import numpy as np
import pickle
import os
import datetime


def pickle_to_ndarray(path, save_path, n_files=120000):
    # Make counter
    n = 0

    # List pickles in 2018/2017 samples
    pickles = os.listdir(path)
    p_len = len(pickles)

    # Create sub-folders so theres 10k in each.
    for i in range(n_files // 10000):
        sub = os.path.join(save_path + 'a' + str(i))
        try:
            os.makedirs(sub)
        except FileExistsError:
            print(sub, 'already exists')

    folders = os.listdir(save_path)
    folder_idx = -1

    # Loop that continues until n_files (def = 120.000) is added
    for i, file in enumerate(pickles):
        if n == n_files:
            break

        infile = open(path + file, 'rb')
        pic = pickle.load(infile)
        infile.close()

        print(f'Pickle loaded: [{i}/{p_len}]  ------  {str(datetime.datetime.now())[11:-7]}')

        for image in pic:

            if n == n_files:
                break

            k_len = np.array(image['image']).shape[0]
            k_wid = np.array(image['image']).shape[1]

            print(k_len,k_wid)

            if (k_len <= 180) and (k_wid <= 80):
                img = image['image']

                # Saving the image
                if n % 10000 == 0:
                    folder_idx += 1
                    if folder_idx != 0:
                        print('folder', folders[folder_idx], 'is full!')

                n += 1
                np.save(save_path + folders[folder_idx] + '/grain' + str(n), img)


path = "M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/validation/Images/2018 samples/"
save_path = "C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/validation_grain/"
pickle_to_ndarray(path, save_path)
