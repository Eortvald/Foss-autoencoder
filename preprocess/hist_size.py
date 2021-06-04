import numpy as np
import matplotlib.pyplot as plt

width = np.load('width_sorted.npy')
height = np.load('heigh_sorted.npy')

#bins = np.linspace(25, 300, 100)

#plt.hist(width, bins, alpha = 0.5, label = 'width')
print(np.percentile(width, [10, 98]))
plt.hist(height, bins = 200, alpha = 0.5, label = 'height')
plt.legend(loc = 'upper right')
plt.savefig('height_distribution')
plt.show()
