import os
import numpy as np

numimages = 869
sizeimages = 265 * 91
i = 0
matrix = np.zeros([numimages, sizeimages * 8])

with os.scandir('C:/Users/Ext1306/Desktop/00') as entries:
    for entry in entries:
        img = np.load(entry)
        matrix[i,:] = img.flatten()
        i += 1
print(np.shape(matrix))
matrix = matrix.T
np.save('pca_matrix', matrix)