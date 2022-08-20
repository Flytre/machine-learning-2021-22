import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier

X, y = make_circles(random_state=100, noise=0.05, n_samples=1000)
plt.scatter(X[:, 0], X[:, 1], s=50, c=y, cmap='viridis')
plt.show()

X_train, X_test = X[:500], X[500:]
y_train, y_test = y[:500], y[500:]

mlp = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=500)
mlp.out_activation_ = 'logistic'

epoch_x = list()
accuracy_y = list()
train_accuracy_y = list()
mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))
for i in range(0, 500):
    mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))
    epoch_x.append(i)
    accuracy_y.append(mlp.score(X_test, y_test))
    train_accuracy_y.append(mlp.score(X_train, y_train))

plt.plot(epoch_x, accuracy_y)
plt.plot(epoch_x, train_accuracy_y)
plt.title("Training/Testing Accuracy over epochs")
plt.show()
