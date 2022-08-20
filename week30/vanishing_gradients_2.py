import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier

X, y = make_circles(random_state=100, noise=0.05, n_samples=1000)
X_train, X_test = X[:500], X[500:]
y_train, y_test = y[:500], y[500:]

layers_x = list()
accuracy_y = list()
for i in range(0, 12):
    layers = tuple([5 for __ in range(0, i)])
    mlp = MLPClassifier(hidden_layer_sizes=layers, activation='relu', max_iter=500, random_state=100)
    mlp.out_activation_ = 'logistic'
    mlp.fit(X_train, y_train)
    layers_x.append(i)
    accuracy_y.append(mlp.score(X_train, y_train))

plt.plot(layers_x, accuracy_y)
plt.title("Accuracy by Number of Layers in Model")
plt.show()
