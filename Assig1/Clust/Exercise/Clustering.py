from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
#question 1
data=pd.read_csv(r"clustering_data.csv",delimiter=",",header=0,index_col=0)
#scaler=preprocessing.StandardScaler().fit(data)
d=data.values.tolist()
X=np.array(d)
kmeans = KMeans(init="random").fit(X)
print(kmeans.labels_)
SSE=kmeans.inertia_
print(SSE)

#question 2
kmeans = KMeans(init="k-means++",n_clusters=8,n_init=100,max_iter=10000,random_state=10).fit(X)
print(kmeans.labels_)
SSE=kmeans.inertia_
print(SSE)