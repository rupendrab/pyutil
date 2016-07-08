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

######

# Use silhouette coefficient to evaluate kmeans

%matplotlib inline
from sklearn import metrics

x1 = np.array([1,2,3,1,5,6,5,5,6,7,8,9,7,9])
x2 = np.array([1,3,2,2,8,6,7,6,7,1,2,1,1,3])
X = np.array(list(zip(x1,x2))).reshape(len(X), 2)

plt.subplots_adjust(left=1, bottom=None, right=3, top=2,
                wspace=None, hspace=None)

plt.subplot(3, 2, 1)
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Instances')
plt.scatter(x1, x2)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']
tests = [2,3,4,5,8]

subplot_counter = 1
for t in tests:
    subplot_counter += 1
    plt.subplot(3, 2, subplot_counter)
    kmeans_model = KMeans(n_clusters=t).fit(X)
    for i, l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l], ls='None')
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.title('K = %s, silhouette coefficient = %.03f' % 
              (t, metrics.silhouette_score(X, 
                                           kmeans_model.labels_,
                                           metric = 'euclidean')))
plt.show()
