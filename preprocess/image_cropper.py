import numpy as np
import os
import datetime

def image_cropper(root_path):
    folders = os.listdir(root_path)
    for i, folder in enumerate(folders):
        print(f'Folder loaded: [{i}/{len(folders)}]  ------  {str(datetime.datetime.now())[11:-7]}')
        with os.scandir(root_path + folder) as entries:
            for entry in entries:
                img = np.load(entry)
                mask = img[:, :, 7]
                height, width = np.shape(img)[0:1]
                if height > 190:
                    for j in range(height):
                        np.sum(mask[j])

