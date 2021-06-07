import numpy as np
import os
i = 0
path = 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/BlobArchive/'
folders = os.listdir(path)
for folder in folders:
    with os.scandir(path + str(folder)) as entries:
        for entry in entries:
            i = i + 1

print(i)