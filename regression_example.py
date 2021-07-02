#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 09:03:10 2021

@author: taterosen

linear regression (ex 5.1, p140)
"""

#Load necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#load dataset into Pandas Dataframe (1st column,year, used as row labels)
df = pd.read_csv('nameOfFile.csv', index_col = 0)

#find correlation btwn two vars in the dataset
print("Correlation Coefficient = ", np.corrcoef(df.Var1, df.Var2)[0,1])

#prepare x and y for the regression model:
y = df.Var1     #response (dependent var)
x = df.Var2     #predictor (indep var)
x = sm.add_constant(x)  #adds a constant term to indep var (bad code tho)

#build regression model using OLS (ordinary least squares)
lr_model = sm.OLS(y,x) .fit()
print (lr_model.summary())

#we pick 100 pts equally spaced from min to max
x_prime = np.linspace(x.Var2.min(), x.Var2.max(), 100)
x_prime = sm.add_constant(x_prime) #add constant

#calculate predicted values
y_hat = lr_model.predict(x_prime)

plt.scatter(x.Var2, y)  #plot raw data
plt.xlabel("Variable 2 - indep")
plt.ylabel("Variable 1 - dep")
# add regression lin, colored in red
plt.plot(x_prime[:, 1], y_hat, 'red', alpha = 0.9)