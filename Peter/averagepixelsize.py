import numpy as np
import os
avsize = 0
with os.scandir('C:/Users/Ext1306/Desktop/00') as entries:
    for entry in entries:
        img = np.load(entry)
        avsize += np.sum(img[:,:,7])

print(avsize/869)