#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 10:41:06 2021

@author: taterosen

hiking data kNN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("hiking.csv")

# Mark ~70% of data for training, use rest for testing
# we'll use 'length_3d', 'uphill', and 'moving_time' features for training a classifier on 'difficulty'
X_train, X_test, y_train, y_test = train_test_split(df[['length_3d', 'uphill', 'moving_time']], df['difficulty'], test_size = .3)
classifier = KNeighborsClassifier(n_neighbors = 50)
classifier.fit(X_train, y_train)

# test classifier by giving it test instances
prediction = classifier.predict(X_test)

# count how many were correctly classified
correct = np.where(prediction == y_test, 1, 0).sum()
print(correct)

# calc accuracy of this classifier
accuracy = correct/len(y_test)
print(accuracy)

#start w array where results (k and corresponding accuracy) will be stored
results = []

for k in range(1, 51, 2):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)
    accuracy = np.where(prediction==y_test, 1, 0).sum() / (len(y_test))
    print("k =", k, "Accuracy =", accuracy)
    results.append([k,accuracy])    #storing the k, accuracy tuple in results array
    
    #convert that series of tuples in a dataframe for easy plotting
    results2 = pd.DataFrame(results, columns=["k","accuracy"])
    
    plt.plot(results2.k, results2.accuracy)
    plt.title("value of k and corresponding classification accuracy")
    plt.show()