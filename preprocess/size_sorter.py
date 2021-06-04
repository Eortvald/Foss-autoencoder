import numpy as np
width = np.load('widths.npy')
height = np.load('height.npy')

width_sorted = np.sort(width)
heigh_sorted = np.sort(height)

np.save('width_sorted', width_sorted)
np.save('heigh_sorted', heigh_sorted)