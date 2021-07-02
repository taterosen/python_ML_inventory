#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 11:29:47 2021

@author: taterosen

regression analysis on rock data
"""

# load numpy and pandas for data manip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load statsmodels as alias 'sm'
import statsmodels.api as sm

# load the data downloaded from twitter
df = pd.read_csv('rock_music.csv')

# look at some correlations
print("correlation coefficient =", 
      np.corrcoef(df.Popularity, df.instrumentalness)[0,1])
print("correlation coefficient =", 
      np.corrcoef(df.speechiness, df.instrumentalness)[0,1])

# create scatterplot of songs' popularity vs energy 
plt.scatter(df.Popularity, df.instrumentalness)

# use regression analysis for predicting popularity(y) using instramentalness (x)
y = df.Popularity #response
x = df.instrumentalness #predictor
x = sm.add_constant(x) #adds a constant term to predictor

lr_model = sm.OLS(y,x).fit()
print(lr_model.summary())

# pick 100 pts equally spaced from min to max
x_prime = np.linspace(x.instrumentalness.min(), x.instrumentalness.max(), 100)
x_prime = sm.add_constant(x_prime) 

# calc predicted values
y_hat = lr_model.predict(x_prime)

plt.figure(1)

plt.subplot(211)
plt.scatter(df.Popularity, df.instrumentalness)

plt.subplot(212)
plt.scatter(x.instrumentalness, y) #plot raw data
plt.xlabel("instrumentalness")
plt.ylabel("Popularity")
plt.plot(x_prime[:,1], y_hat, 'blue') # add red regression line
