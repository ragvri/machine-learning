# actually predicting data using linear regression

import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

style.use('ggplot')

quandl.ApiConfig.api_key = 'vhkrsKz3TmUN6Qa4QjZK'
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_percent'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['percent_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_percent', 'percent_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna('-9999', inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
X = np.array(df.drop(['label'], 1))  # X is an array of features. Everything other than 'Label' is a feature
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
Y = np.array(df['label'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)

forecast_set = clf.predict(X_lately)  # predicts the value for an array or a single value using our classifier
print(forecast_set, accuracy, forecast_out)

# plotting our predictions

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]  # .loc referes the index by name, iloc by
    #  number

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
