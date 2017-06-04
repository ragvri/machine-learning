import numpy as np
from sklearn.cluster import MeanShift, KMeans
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt

'''
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
'''
pd.options.mode.chained_assignment = None  # default='warn'
# https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
df = pd.read_excel('titanic.xls')

original_df = pd.DataFrame.copy(df)
df.drop(['body', 'name'], 1, inplace=True)
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
df.drop(['ticket', 'home.dest'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)
labels = clf.labels_
cluster_centers = clf.cluster_centers_
n_clusters_ = len(np.unique(labels))
original_df['cluster_group'] = np.nan
for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]

survival_rates = {}

for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    survival_cluster = temp_df[(temp_df['survived'] == 1)]

    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)
print(original_df[(original_df['cluster_group'] == 2)])
