"""
Classifying with svm when more than 2 groups:
1) OVR : one vs rest:
   separate one group from rest of the data
2) OVO : One vs One:
   assume 3 groups : 1,2,3
   first make hyperplane for 1 vs 2 and 1 vs 3
   then 2 vs 3
"""

# check the documentation for svm.SVM
import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

df = pd.read_csv("breast-cancer-wisconsin.data.txt")

df.replace('?', -9999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))  # here 1 means we want to drop the columns
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC()  # SVC: support vector classifier
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
example_measures = example_measures.reshape(1, -1)
prediction = clf.predict(example_measures)
print(prediction)
