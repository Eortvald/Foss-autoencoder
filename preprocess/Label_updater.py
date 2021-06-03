import pandas as pd
import numpy as np
import os

path = 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/BlobArchive/'
labels_updated = pd.read_csv('labels.csv', error_bad_lines = False, index_col = False, dtype = 'unicode')
labels_updated.insert(loc = 8, column = "ccolor", value = str(), allow_duplicates = False)
labels_updated = labels_updated.set_index(['Names'])

folders = os.listdir(path)

n = 0
for folder in folders:
    n += 1
    print(folder, n)
    with os.scandir(path + str(folder)) as entries:
        for entry in entries:
            img = np.load(entry)
            if img[0,0,2] == 255:
                labels_updated.loc[(entry.name.split(".")[0]),"ccolor"] = "Green"

            elif img[0,0,4] == 255:
                labels_updated.loc[(entry.name.split(".")[0]),"ccolor"] = "Red"

            elif img[0,0,1] == 255:
                labels_updated.loc[(entry.name.split(".")[0]),"ccolor"] = "Blue"

labels_updated.to_csv('labels_updated.csv', index = False)
