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