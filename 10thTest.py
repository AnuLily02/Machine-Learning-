from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
X, _ = make_blobs(n_samples=1000, centers=3, n_features=2)
print("Features :")
print(X)
kmeans = KMeans(n_clusters= 3, random_state=0, n_init="auto").fit(X)
print("data points belongs to clusters ",kmeans.labels_)
print("cluster centroids are as follows",kmeans.cluster_centers_)

pred, _ = make_blobs(n_samples=1, centers=1, n_features=2)
print("new data points",pred)
print("new clusters belong to ",kmeans.predict(pred))
plt.scatter(X[:, 0], X[:, 1], alpha=0.1,c=kmeans.labels_, cmap='viridis')
plt.scatter(pred[:,0],pred[:,1],marker='x',c='black',s=100,label='New datapoint')

for i,center in enumerate(kmeans.cluster_centers_):
    plt.text(center[0],center[1],f'Cluster {i}',fontsize=12,color='red',ha='center')
plt.show()
