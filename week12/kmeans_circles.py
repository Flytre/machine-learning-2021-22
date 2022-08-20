from sklearn.datasets import make_circles
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, y = make_circles(random_state=100, noise=0.05)
kmeans = KMeans(2, random_state=1434)
labels = kmeans.fit_predict(X)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="+", s=150)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

plt.scatter(X[:, 0], X[:, 1], s=50, c=y, cmap='viridis')
plt.show()
