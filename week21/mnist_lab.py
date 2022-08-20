import ssl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

ssl._create_default_https_context = ssl._create_unverified_context

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

mlp = MLPClassifier(hidden_layer_sizes=(50,))

epoch_x = list()
loss_y = list()
accuracy_y = list()
mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))
for i in range(0, 100):
    # whole forward and backward pass in here, along with error calculation
    mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))
    epoch_x.append(i)
    accuracy_y.append(mlp.score(X_test, y_test))
    loss_y.append(mlp.loss_)

plt.plot(epoch_x, loss_y)
plt.title("loss")
plt.show()

plt.plot(epoch_x, accuracy_y)
plt.title("accuracy")
plt.show()
