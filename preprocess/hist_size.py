import numpy as np
import matplotlib.pyplot as plt

pic_width = np.load('pickle_width.npy')
pic_height = np.load('pickle_height.npy')

height = np.load('height.npy')
width = np.load('widths.npy')


#bins = np.linspace(25, 300, 100)

#plt.hist(width, bins, alpha = 0.5, label = 'width')
print(np.percentile(pic_width, np.linspace(90,99,10)), 'picke width')
print(np.percentile(pic_height, np.linspace(90,99, 10)), 'pickle height')
print(np.percentile(width, np.linspace(90,99,10)), 'width')
print(np.percentile(height, np.linspace(90,99, 10)), 'height')

plt.hist(pic_height, bins = 200, alpha = 0.5, label = 'height')
# plt.scatter(x = pic_height, y = pic_width, marker = '+', alpha = 0.002)
plt.legend(loc = 'upper right')
plt.savefig('C:/Users/Nullerh/Documents/DTU_SCHOOL_WORK/Semester4/02466_Project/KORN/plots/preprocess-plots/pic_height')

plt.hist(pic_width, bins = 200, alpha = 0.5, label = 'width')
plt.legend(loc = 'upper right')
plt.savefig('C:/Users/Nullerh/Documents/DTU_SCHOOL_WORK/Semester4/02466_Project/KORN/plots/preprocess-plots/pic_width')

