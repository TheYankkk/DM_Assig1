from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

#create a numpy array
X = np.array([[0, 0], [0, 1/2],[1, 1/2], [1, 1],[3, 0], [3, 1], [4, 1]])
x=np.array([0,0,1,1,3,3,4])
y=np.array([0,1/2,1/2,1,0,1,1])

orignal= np.array([[0, 0], [1, 1],[2.5,0.8]])
kmeans = KMeans(init=orignal, n_clusters=3, max_iter=10000, n_init=100).fit_predict(X)
center = KMeans(init=orignal, n_clusters=3, max_iter=10000, n_init=100).fit(X)
plt.scatter(x,y,c=kmeans)
for center in center.cluster_centers_:
    plt.scatter(center[0], center[1], marker="p", edgecolors="red")
plt.show()
print(kmeans.cluster_centers_)