#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 11:01:30 2021

@author: taterosen

hiking clustering
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import style class from matplotlib and use to apply ggplot styling
from matplotlib import style
style.use("ggplot")

# get KMeans class from clustering lib avail within scikit-learn
from sklearn.cluster import KMeans

# define data points on 2D plane using cartesian coordinates
X = pd.read_csv("hiking.csv", usecols=(6,7), skiprows=0)

# perform clustering using k-means alg
kmeans = KMeans (n_clusters = 4)
kmeans.fit(X)

# 'kmeans' holds the model; extract info abt clusters as rep by their centroids, along w their labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# define colors array
colors = ["g.", "r.", "c.", "y."]

# loop to go through each data pt, plotting it on the plane w/ color from above list(1 color per cluster)

for i in range (len(X)):
    print("coordinate:", X.iloc[i], "Label:", labels[i])
    plt.plot (X.iloc[i][0], X.iloc[i][1], colors[labels[i]], markersize = 10)

# plot the centroids using "x"
plt.scatter(centroids[:,0], centroids[:,1], marker = "x", s = 150, linewidths = 2, zorder = 10)
plt.ylabel("moving time")
plt.xlabel("uphill")
plt.show()
