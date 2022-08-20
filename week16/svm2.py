import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

data = pd.read_csv("Iris.csv")

X = np.empty(shape=(0, 2))
y = np.array([])

keys = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

for row in data.iterrows():
    arr = row[1].array
    X = np.vstack([X, np.array([arr[2], arr[3]])])
    y = np.append(y, keys[arr[4]])

h = .02
clf = svm.SVC(kernel='poly', C=1).fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

plt.subplots_adjust(wspace=0.4, hspace=0.4)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.get_cmap("inferno"), alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.get_cmap("viridis"))
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
