# Import libraries
import numpy as np
from sklearn.cluster import KMeans

# Dataset
X = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11]
])

# Create model with 2 clusters
kmeans = KMeans(n_clusters=2)

# Train model
kmeans.fit(X)

# Cluster labels
print("Cluster labels:", kmeans.labels_)

# Centroids
print("Centroids:", kmeans.cluster_centers_)
