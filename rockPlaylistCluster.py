#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:11:45 2021

@author: taterosen

spotify rock playlist clustering
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import style class from matplotlib and use to apply ggplot styling
from matplotlib import style
style.use("ggplot")

# get KMeans class from clustering lib avail within scikit-learn
from sklearn.cluster import KMeans

X = pd.read_csv("rock_music.csv", usecols=(2,5), skiprows=0)

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
plt.xlabel('popularity')
plt.ylabel('danceability')
plt.show()
