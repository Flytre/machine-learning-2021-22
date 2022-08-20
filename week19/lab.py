import math
import random

import matplotlib.pyplot as plt
import numpy as np


def perceptron(activation, weight, bias, x):
    return np.vectorize(activation)((x @ weight) + bias)


def step(t: int):
    return 1 if t > 0 else 0


def sigmoid(t: float):
    return 1 / (1 + math.e ** -t)

# LINE FUNCTION!
def matches(x, y):
    return y > x * 2 + 1


def create_points():
    points = list()
    x = list()
    y = list()
    for i in range(0, 50):
        x.append((random.uniform(-10, 10)))
        y.append((random.uniform(-10, 10)))
        points.append((x[i], y[i]))
    return points, x, y


def train(points2: list, learning_rate=0.005, activation=sigmoid):
    weight = np.array([[random.random()]])
    bias = np.array([[random.random()]])
    print("initial weight = ", weight, "initial bias = ", bias)
    while True:
        accuracy = 0
        changed = False
        for i in range(0, len(points2)):
            x = points2[i][0]
            y = points2[i][1]
            expected = 1 if matches(x, y) else 0
            error = expected - perceptron(activation, weight, bias, np.array([[x]]))[0, 0]
            if error == 1:
                changed = True
            else:
                accuracy += 1
            weight = weight + learning_rate * error * np.array([[x]])
            bias = bias + learning_rate * error
        if not changed or accuracy / len(points2) >= 0.96:
            break
    print("weight = ", weight, "bias =", bias, "accuracy minimum = 96%")


points, xl, yl = create_points()
plt.plot(xl, yl, "ro")
plt.plot([-10, 10], [-19, 21])
plt.show()
train(points)
