import numpy as np
import os

path = 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/BlobArchive/'
folders = os.listdir(path)
widths = np.array([])
heights = np.array([])
for folder in folders:
    print(folder)
    with os.scandir(path + str(folder)) as entries:
        for entry in entries:
            img = np.load(entry)
            widths = np.append(widths, np.shape(img)[1])
            heights = np.append(heights, np.shape(img)[0])
np.save('widths.npy')
np.save('height.npy')
