"""
Supervised: We have told the machines what the classes were

Clustering:
1) Flat
2) Hirierchal

In both, The machine is just given the featureset. Then the machine itself searches for groups or clusters.

With Flat Clustering, we tell the machine to find 2 clusters or 3 clusters.
With Hirierchal clusterign , the machine figures out how many groups are there

First Algo we use:
1) K Means : K is the number of clusters we want -> does Flat Clustering
2) Mean Shift : Hirierchal Clustering

K mean working:
Chose K centroids randomly in the beginning, mostly the first k points are taken.
Calculate the distance of each featureset to the centroids and classify each accordingly.
Then take all the featureset of one cluster and take mean of those. These are the new centroids.
Repeat until the centroids are no longer moving.

Downside of K means: It always tries to find same sized groups

"""

import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
import numpy as np

style.use('ggplot')

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]]
             )


clf = KMeans(n_clusters=2)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ["g.", "r.", "c.", "b.", "k.", "y."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=20)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150)
plt.show()
