#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 10:31:32 2021

@author: taterosen

clustering example 5.4, p150
"""

import numpy as np
import matplotlib.pyplot as plt

# import style class from matplotlib and use to apply ggplot styling
from matplotlib import style
style.use("ggplot")

# get KMeans class from clustering lib avail within scikit-learn
from sklearn.cluster import KMeans

# define data points on 2D plane using cartesian coordinates
X = np.array ([ [1,2],
                [5,8],
                [1.5,1.8],
                [8, 8],
                [1, 0.6],
                [9, 11]])

# perform clustering using k-means alg
kmeans = KMeans (n_clusters = 3)
kmeans.fit(X)

# 'kmeans' holds the model; extract info abt clusters as rep by their centroids, along w their labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# define colors array
colors = ["g.", "r.", "c.", "y."]

# loop to go through each data pt, plotting it on the plane w/ color from above list(1 color per cluster)
for i in range (len(X)):
    print("coordinate:", X[i], "Label:", labels[i])
    plt.plot (X[i][0], X[i][1], colors[labels[i]], markersize = 10)
    
# plot the centroids using "x"
plt.scatter(centroids[:,0], centroids[:,1], marker = "x", s = 150, linewidths = 2, zorder = 10)
plt.show()