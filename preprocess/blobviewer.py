import os
import matplotlib.pyplot as plt
import numpy as np
i = 0


path = 'C:/ASB/Projects/EyefossAutoencoder/Fagprojekt-2021/tenkblobs'
with os.scandir(path) as entries:
    for entry in entries:
        i += 1
        img_labeled = np.load(entry, allow_pickle = True)
        img = img_labeled[0]
        rgb = np.dstack((img[:,:,4], img[:,:,2], img[:,:,1]))
        plt.imshow(rgb)
        plt.title(img_labeled[1])
        plt.show()
