import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC

centers = [(1, 4), (20, 1)]
cluster_std = [1.5, 1.5]
X, y = make_blobs(n_samples=400, random_state=99, centers=centers, cluster_std=cluster_std)
plt.figure(figsize=(10, 5))

clf = LinearSVC(C=1, loss="hinge", random_state=42).fit(X, y)
decision_function = clf.decision_function(X)
support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
support_vectors = X[support_vector_indices]

#plot the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)


axes = plt.gca()  # get the axes
xlim = axes.get_xlim()  # x bounds
ylim = axes.get_ylim()  # y bounds

#create a grid of x values and y values
xx, yy = np.meshgrid(
    np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50)
)
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


#plot the lines
plt.contour(
    xx,
    yy,
    Z,
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
)

plt.show()
