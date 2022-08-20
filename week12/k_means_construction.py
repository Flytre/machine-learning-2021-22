from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

centers = [(0, 0), (6, 1.5), (6, -1.5)]
cluster_std = [1.1, 0.3, 0.3]
X, y = make_blobs(n_samples=400, random_state=99, centers=centers, cluster_std=cluster_std)
print(X)
print(y)
# kmeans = KMeans(3, random_state=45235)
# labels = kmeans.fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=20, cmap='viridis')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="+", s=150)
# plt.show()
