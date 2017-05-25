"""
classification algo. Divides the data into groups
Given pluses and minuses in a graph, and an unknown point would it belong to pluses or minuses

Linear regression aim was to create a model that best fits the data

We would classify depending upon how close it lies to one group. This is essentialy the nearest neighbours.

With k nearest neighbours , eg k =2. We check only the 2 nearest neigbours and then decide which group it lies based
on that. If both the neighbours in same group well enough. Else if split vote, So we take odd number of ks. For
three groups min value of k is 5.

If 2 groups and k =3 and we get 2 votes for 1 group, then confidence =2/3

Downfalls: we need to calculate distance from all. So on huge datasets it would be a problem.
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

df = pd.read_csv("breast-cancer-wisconsin.data.txt")

df.replace('?', -9999, inplace=True)  # replacing missing values. Most algo recognise -9999 as outlier
df.drop(['id'], 1, inplace=True)
# print(df.head())

X = np.array(df.drop(['class'], 1))  # here 1 means we want to drop the columns
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier(n_jobs=-1)   # default k value is 5. By setting n_jobs = -1, we are threading
# the classifier. This allows us to run the classifier on multiple testst simultaneously
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])  # this returns 9 rows
example_measures = example_measures.reshape(1, -1)  # -1 means unspecified number of cols
prediction = clf.predict(example_measures)
print(prediction)

"""
Instead of using k nearest neighbors, we could get the best fit line for each of the groups using linear regression
and then find the distance of our point from the lines thus obtained. The group corresponding to the line having the
least distance will be the answer.

However if the data is non linear, then best fit line won't work. But k nearest neighbors will work
"""