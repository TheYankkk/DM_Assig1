from sklearn.cluster import KMeans
import numpy as np

#create a numpy array
X = np.array([[0, 0], [0, 1/2],[1, 1/2], [1, 1],[4, 0], [4, 1], [5, 1]])
orignal= np.array([[0, 0], [1, 1]])
kmeans = KMeans(init=orignal, n_clusters=2, max_iter=10000, n_init=100).fit(X)

print(kmeans.cluster_centers_)