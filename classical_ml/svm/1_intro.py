"""
SVM is a binary classifier: Separates only in 2 groups at a time. That does not mean it can only "classify"
in 2 groups. It just means that at a time it can only separate one group from the rest.

The 2 groups are denoted as positives and negatives.

We want a street that separates the 2 goups and is as wide as possible. Then we consider a vector(w) which is
perpendicular to the street's median. Now to check if a point lies on one side or other, we take the dot product of the
unknown point (vector u) with (vector w).  Now from this length we can check if the point is on left or the right side
of the street
OR
(vector u).(vector w) + b >=0        (1)
 then one side else on other
Here we don't know w and b, we just know w is perpendicular
ALSO
(X+).(vector w) +b > =1 where X+ is the positive sample
and
(X-).(vector w) + b<=1 where X- is the negative sample.

All these are constraints

We introduce Yi such that Yi = 1 for +
                          Yi = -1 for _
Multiplying both by Yi gives
                            Yi* [(X).(vector w) + b] -1 >=0      (2)
where X is any known sample
Also for Xi in the gutter (2) = 0 These are called Support Vectors

Q) How to find the widht of the "street"?
A) If we had a unit normal to the "gutter", then the dot product (X+ - X-).(w/|w|)   (3)
   is width as w is a normal to the street. This can be simplified using (2)
   The width comes out as : 2/|w|

To max the widht of the street, we want to min |w| or
min 1/2(|w|)^2    (4)

To solve a question having to find extremeties and given some constraints, we use Lagrange. We find that
w = sum(Ci*Xi*Yi)  (5)
sum(Ci*Yi) = 0   (6)

(vector X).(vector W) + bias = 0 gives decision boundary

It finds the decision boundry which is a  boundry which separates the 2 groups.

"""
# applying the svm lib on the breast cancer eg

import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

df = pd.read_csv("breast-cancer-wisconsin.data.txt")

df.replace('?', -9999, inplace=True)  # replacing missing values. Most algo recognise -9999 as outlier
df.drop(['id'], 1, inplace=True)
# print(df.head())

X = np.array(df.drop(['class'], 1))  # here 1 means we want to drop the columns
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC()  # SVC: support vector classifier
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])  # this returns 9 rows
example_measures = example_measures.reshape(1, -1)  # -1 means unspecified number of cols
prediction = clf.predict(example_measures)
print(prediction)
