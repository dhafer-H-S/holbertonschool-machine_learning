#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Assuming lib is defined correctly

data = np.load('data.npy')
labels = np.load('labels.npy')

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
ax.set_title('PCA of Iris Dataset')

colors = ['blue', 'red', 'yellow']
label_colors = [colors[int(label)] for label in labels]

ax.scatter(pca_data[:,0], pca_data[:,1], pca_data[:,2], c=label_colors, cmap='plasma')

plt.show()
