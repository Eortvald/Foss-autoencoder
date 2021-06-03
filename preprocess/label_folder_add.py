import pandas as pd
import numpy as np
import os

path = 'M:/R&D/Technology access controlled/Projects access controlled/AIFoss/Data/BlobArchive/'
labels_updated = pd.read_csv('labels.csv', error_bad_lines = False, index_col = False, dtype = 'unicode')
labels_updated.insert(loc = 8, column = "folder", value = str(), allow_duplicates = False)
labels_updated = labels_updated.set_index(['Names'])

folders = os.listdir(path)

for folder in folders:
    with os.scandir(path + str(folder)) as entries:
        for entry in entries:
            labels_updated.loc[(entry.name.split(".")[0]), "folder"] = str(folder)
labels_updated.to_csv('labels_folder.csv', index = False)
