import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pandas as pd

style.use('ggplot')

"""
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
"""

df = pd.read_excel('titanic.xls')

# here we find that some values are non numeric
# eg sex column, we take the set of sex column and then assign then numbers

df.drop(['body', 'name'], 1)
df.apply(pd.to_numeric, errors='ignore')
df.fillna(0, inplace=True)


def handle_non_numerical_data(df):
    le = LabelEncoder()
    columns = list(df.columns.values)  # to handle non numeric data types use LabelEncoder()

    for column in columns:
        l = []
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            for i in df[column]:
                l.append(i)
            le.fit(np.array(l))
            x = le.transform(l)
            df[column] = x
    return df


df = handle_non_numerical_data(df)

# once the clusters are obtained,we could svm etc

X = np.array(df.drop(['survived'], 1)).astype(float)
X = preprocessing.scale(X)  # important
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)
labels = clf.labels_
correct = 0
for i in range(len(X)):
    predict_me = np.array(
        X[i].astype(float))  # the first centroid is 0. It might be that the survived is 1 and we get 0 because
    # that is the first centroid. So our accuracy would be 20% instead of 80%
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction == y[i]:
        correct += 1

print(correct / len(X))
