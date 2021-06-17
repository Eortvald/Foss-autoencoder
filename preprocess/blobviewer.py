import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
path = 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/BlobArchive_v2/'
folders = os.listdir(path)
i = 0
fig, ar = plt.subplots(1, 3, sharey=True)
for folder in folders:
    with os.scandir(path + folder) as entries:
        for entry in entries:
            img = np.load(entry)
            h, w = np.shape(img)[0:2]
            if (h > 450):
                rgb = np.dstack((img[:,:,4], img[:,:,2], img[:,:,1]))
                ar[i].imshow(rgb)
                i += 1
                if i == 3:
                    plt.show()
