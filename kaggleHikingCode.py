#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 14:48:57 2021

@author: taterosen

hiking kaggle code - "predicting hike times"
"""

import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pandas as pd
from matplotlib import pyplot

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor

"""
from keras import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
"""
import numpy as np



df = pd.read_csv('hiking.csv')
#print(df.head(n=2))

##add in an average speed column
df['avg_speed'] = df['length_3d']/df['moving_time']

##difficulty rating changed to a numeric value for easier processing
df['difficulty_num'] = df['difficulty'].map(lambda x: int(x[1])).astype('int32')
#print(df.head(n=2))

#print(df.describe())

## drop null values/extremes
df.dropna()
df = df[df['avg_speed'] < 2.5] # an avg of > 2.5m/s is probably not a hiking activity


def retain_values(df, column, min_quartile, max_quartile):
    q_min, q_max = df[column].quantile([min_quartile, max_quartile])
    #print("Keeping values between {} and {} of column {}".format(q_min, q_max, column))
    return df[(df[column] > q_min) & (df[column] < q_max)]

# drop elevation outliers
df = retain_values(df, 'min_elevation', 0.01, 1)

##show grid of correlations between columns
print(df.corr())

"""
##use a linear regression model to predict the moving time based on 
    ##the most correlated fields (length_3d, uphill, downhill and max_elevation)
y = df.reset_index()['moving_time']
x = df.reset_index()[['downhill', 'uphill', 'length_3d', 'max_elevation']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

lasso = Lasso()
lasso.fit(x_train, y_train)
print("Coefficients: {}".format(lasso.coef_))

y_pred_lasso = lasso.predict(x_test)

r2 = r2_score(y_test, y_pred_lasso)
mse = mean_squared_error(y_test, y_pred_lasso)

print("r2:\t{}\nMSE: \t{}".format(r2, mse))

##Gradient Boosting Regressor?
gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)
y_pred_gbr = gbr.predict(x_test)

r2 = r2_score(y_test, y_pred_gbr)
mse = mean_squared_error(y_test, y_pred_gbr)

print("r2:\t{}\nMSE: \t{}".format(r2, mse))


##Regression with Keras

model = Sequential()
model.add(Dense(12, input_shape=(4,)))
model.add(Dense(5, input_shape=(4,)))
model.add(Dense(1))
model.compile(optimizer=Adam(0.001), loss='mse')    ##Using TensorFlow backend.

model.load_weights(filepath='./keras-model.h5')
y_pred_keras = model.predict(x_test)

r2 = r2_score(y_test, y_pred_keras)
mse = mean_squared_error(y_test, y_pred_keras)

print("r2:\t{}\nMSE: \t{}".format(r2, mse))


##Ensemble Results
combined = (y_pred_keras[:,0] + y_pred_gbr * 2) / 3.0
r2 = r2_score(y_test, combined)
mse = mean_squared_error(y_test, combined)

print("r2:\t{}\nMSE: \t{}".format(r2, mse))

c = pd.DataFrame([combined, y_pred_keras[:,0], y_pred_lasso, y_pred_gbr, y_test]).transpose()
c.columns = ['combined', 'keras', 'lasso', 'tree', 'test']
c['diff_minutes'] = (c['test'] - c['combined']) / 60
c.describe()
"""


















