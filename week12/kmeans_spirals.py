import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

N = 400
theta = np.sqrt(np.random.rand(N)) * 2 * pi

r_a = 2 * theta + pi
data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
x_a = data_a + np.random.randn(N, 2)

r_b = -2 * theta - pi
data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
x_b = data_b + np.random.randn(N, 2)

res_a = np.append(x_a, np.zeros((N, 1)), axis=1)
res_b = np.append(x_b, np.ones((N, 1)), axis=1)

res = np.append(res_a, res_b, axis=0)
np.random.shuffle(res)

plt.scatter(x_a[:, 0], x_a[:, 1])
plt.scatter(x_b[:, 0], x_b[:, 1])
plt.show()

combined = np.concatenate((x_a, x_b))
kmeans = KMeans(2, random_state=3434)
labels = kmeans.fit_predict(combined)
plt.scatter(combined[:, 0], combined[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="+", s=150)
plt.show()
