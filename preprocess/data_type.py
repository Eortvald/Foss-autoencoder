import numpy as np
import os

path = 'temp/'
with os.scandir(path) as entries:
    for entry in entries:
        img = np.load(entry)
        for i in range(np.shape(img)[2]):
            print(img[:,:,i].dtype)