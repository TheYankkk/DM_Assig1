from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv(r"clustering_data.csv",delimiter=",",header=0,index_col=0)
d=data.values.tolist()
X=np.array(d)
kmeans = KMeans(init="random").fit(X)
print(kmeans.labels_)

