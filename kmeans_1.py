import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(3.5, 4.5, (2, 10))
# print(cluster1) 2 x 10 matrix 

X = np.hstack((cluster1, cluster2)).T
# print(X)
# np.vstack(X).T

K = range(1,10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    # if (k == 2):
        # print(kmeans.cluster_centers_)
        # print(cdist(X, kmeans.cluster_centers_, 'euclidean'))
        # print(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1))
        # print(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]) 
        # print(X.shape[0])
    meandistortions.append(
        sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) /
        X.shape[0]
    )


%matplotlib inline
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the elbow method')
plt.show()

######

kmeans_2 = KMeans(n_clusters=2)
kmeans_2.fit(X)

######

labels = kmeans_2.labels_
# print(X)
x = [x[0] for x in X]
y = [x[1] for x in X]
X_transposed = X.T
x = X_transposed[0]
y = X_transposed[1]
plt.scatter(x, y, c=labels.astype(np.int))
plt.show()
