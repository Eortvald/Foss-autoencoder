import numpy as np
import os
import pandas as pd

def seventy_k_prep(path, save_path, n_files = 70000):
    #Make counter
    n = 0
    drop_list = []

    #Create sub-folders so theres 10k in each.
    for i in range(n_files//10000):
        sub = os.path.join(save_path + 'b' + str(i))
        try:
            os.makedirs(sub)
        except FileExistsError:
            print(sub, 'already exists')

    folders = os.listdir(save_path)
    folder_idx = -1

    df = pd.read_csv('Classifier_labels.csv')
    for i in range(len(df)):

        img_name = df.iloc[i]['Names'] + '.npy'
        img_folder = df.iloc[i]['folder'] + '/'
        img = np.load(path + img_folder + img_name)

        # Apply mask
        mask = img[:, :, 7]
        img = np.where(mask[..., None] != 0, img, [0, 0, 0, 0, 0, 0, 0, 0])

        # Trim/Crop image
        img = np.delete(img, np.where(np.sum(mask, axis=1) == 0)[0], axis=0)
        h = np.shape(img[:, :, 7])[0]
        img = np.delete(img, np.where(np.sum(mask, axis=0) == 0)[0], axis=1)
        w = np.shape(img[:, :, 0])[1]

        if (w < 81) and (h < 181):
            if n % 10000 == 0:
                folder_idx += 1
            np.save(save_path + folders[folder_idx] + '/' + img_name)
            n += 1
        else:
            drop_list.append(i)

    df.drop(drop_list)
    if len(df) < 70000:
        print('NOT ENOUGH IMAGES')
    df.to_csv('seventyk_labels.csv')


path = 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/BlobArchive_v2/'
save_path = 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/'
seventy_k_prep(path, save_path)