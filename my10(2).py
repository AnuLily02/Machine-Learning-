from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate synthetic data
X, _ = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=0)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)

# Predict cluster for a new data point
new_point, _ = make_blobs(n_samples=1, centers=1, n_features=2, random_state=1)
new_cluster = kmeans.predict(new_point)

# Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5)
plt.scatter(new_point[:, 0], new_point[:, 1], marker='x', c='black', s=100, label='New data point')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='*', label='Centroids')

# Annotate cluster centers
for i, center in enumerate(kmeans.cluster_centers_):
    plt.text(center[0], center[1], f'Cluster {i}', fontsize=12, color='red', ha='center')

plt.legend()
plt.show()
