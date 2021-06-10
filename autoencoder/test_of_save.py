import numpy as np
import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'relative/path/to/file/you/want')

print(dirname)
arr = np.random.randint(2, size=10)

#np.save('./data/test', arr)
