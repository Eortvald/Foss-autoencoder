import numpy as np
import matplotlib.pyplot as plt

width = np.load('pickle_width.npy')
height = np.load('pickle_height.npy')

#bins = np.linspace(25, 300, 100)

#plt.hist(width, bins, alpha = 0.5, label = 'width')
print(np.percentile(width, np.linspace(90,99,10)))
print(np.percentile(height, np.linspace(90,99, 10)))
'''
plt.hist(width, bins = 200, alpha = 0.5, label = 'width')
plt.legend(loc = 'upper right')
plt.savefig('width_distribution')
plt.show()
'''