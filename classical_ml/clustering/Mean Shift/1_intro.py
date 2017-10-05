"""
It is hirirechal clustering. Automatically figures out the number of clusters needed and where those clusters are.

Here we say every featureset is a cluster center.

It has something called Radius/Bandwidth. Every data point has a circle of radius or a bandwidth around it.
Then we take the mean of data points in a bandwidth.Then this would have a new bandwidth. Repeat till when centroid does
not move.

Repeat for other data points. 2 Centroids from different bandwidths could coincide.

"""
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import MeanShift
from mpl_toolkits.mplot3d import Axes3D

style.use('ggplot')

centres = [[1, 1, 1], [5, 5, 5], [3, 10, 10]]
X, _ = make_blobs(n_samples=100, centers=centres, cluster_std=1)

ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

colors = ['r', 'g', 'b', 'c', 'k', 'y']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker='x', color='k', s=150,
           linewidths=5)
plt.show()
