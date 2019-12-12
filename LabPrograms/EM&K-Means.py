#IMport modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

#Reading of Dataset
data = pd.read_csv('EMK.csv')
print("Input Data and Shape")
print(data.shape)
data.head()

#Extraction of Data
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))

#Graph Plot of the Dataset
print("X   ", X)
print('Graph for whole dataset')
plt.scatter(f1, f2, c='blue', s=7)
plt.show()

#KMeans Clustering Algorithm
kmeans = KMeans(3, random_state=0)

#Kmeans labels
labels = kmeans.fit(X).predict(X)
print("labels    ",labels)

#KMeans Cluster Centroids
centroids = kmeans.cluster_centers_
print("centroids    ",centroids)

#KMeans Graph
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
print('Graph using Kmeans Algorithm')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
plt.show()

#EM Algorithm
gmm = GaussianMixture(n_components=3).fit(X)
labels = gmm.predict(X)
probs = gmm.predict_proba(X)
size = 10 * probs.max(1) ** 3

#EM Graph
print('Graph using EM Algorithm')
plt.scatter(X[:, 0], X[:, 1], c=labels, s=size, cmap='viridis')
plt.show()



