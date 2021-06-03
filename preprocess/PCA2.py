import numpy as np
import sklearn
from sklearn.decomposition import PCA

matrix = np.load('pca_matrix.npy')
#The Sklearn way
pca = PCA(n_components=3,whiten=True)
pca.fit(matrix)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
'''
#The manual way
for i in range(8):
    if i == 7:
        matrix[i] = matrix[i] - np.mean(matrix[i])
        break
    matrix[i] = (matrix[i] - np.mean(matrix[i]))/np.std(matrix[i])
cov_matrix = np.cov(matrix)
eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
ratio = eigenvals[0]**2/(np.dot(eigenvals, eigenvals))
print(ratio)
'''