import math

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import random


def perceptron(activation, weight, bias, x):
    return np.vectorize(activation)((x @ np.transpose(weight)) + bias)


def step(t: int):
    return 1 if t > 0 else 0


def sigmoid(t: float):
    return 1 / (1 + math.e ** -t)


dataset = datasets.load_iris(return_X_y=False, as_frame=False)
X = dataset.data[0:100]
y = dataset.target[0:100]

# display Iris data as 2d projection of sepal traits
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Actual categories')
plt.show()

# train neural network code
# architecture: 4 inputs --> 1 perceptron -> 1 output

weights = np.random.rand(1, 4)
bias = random.random()
learning_rate = 10
print("initial weights = ", weights, "initial bias = ", bias)

# train model
predicted = [0] * 100
accuracy = 0
for epoch in range(0, 100):
    accuracy = 0
    changed = False
    for i in range(0, len(X)):
        inputs = X[i]
        expected = y[i]
        received = perceptron(step, weights, bias, inputs)[0]
        error = expected - received
        if error == 0:
            accuracy += 1
        predicted[i] = received
        weights = weights + learning_rate * error * inputs
        bias = bias + learning_rate * error

# display Iris data as 2d projection of sepal traits
plt.scatter(X[:, 0], X[:, 1], c=predicted, s=20, cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Predicted Categories [Accuracy = ' + str(accuracy) + '%]')
plt.show()
