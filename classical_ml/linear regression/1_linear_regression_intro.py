# applying linear regression to a training data and checking its accuracy against a test data using inbuilt libraries

import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# using preprocessing for scaling. We want the features between -1 to 1. Helps to increase accuracy and processing speed
# train-test-split is used to get the training and testing samples.
# svm is used to do regression

quandl.ApiConfig.api_key = 'vhkrsKz3TmUN6Qa4QjZK'
df = quandl.get('WIKI/GOOGL')

# print(df.head())  # the open, high low are features of the stock. But we need meaningul features
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_percent'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['percent_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_percent', 'percent_change', 'Adj. Volume']]
# print(df.head())

# features are used to predict the label.

forecast_col = 'Adj. Close'
df.fillna('-9999', inplace=True)  # fills the nan data with -9999

forecast_out = int(math.ceil(0.01 * len(df)))  # we will be predicting the final 10% of the data

# let forecast out be = 10 days
df['label'] = df[forecast_col].shift(-forecast_out)  # our label here is the stock price 10 days into the future. So
# based on historical data we want to predict the stock close 10 days into the future. For that, we create a new
# column for the label and shift the Adj. close 10 days up

df.dropna(inplace=True)  # removing the rows whose label value we don't know, we will predict this by regression
# print(df.tail())

X = np.array(df.drop(['label'], 1))  # X is an array of features. Everything other than 'Label' is a feature
X = preprocessing.scale(X)

Y = np.array(df['label'])

# print(len(X), len(Y))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)  # shuffles X and Y and outputs X_train,
# Y_train

clf = LinearRegression()
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)  # testing the accuracy of the classifier
# train and test always on different data
print(accuracy)  # the accuracy will be the squared error done in numerical analysis


