import numpy as np


def perceptron(activation, weight, bias, x):
    return np.vectorize(activation)((x @ np.transpose(weight)) + bias)


def step(t: int):
    return 1 if t > 0 else 0


def func_and(input):
    return perceptron(step, np.array([[1, 1]]), np.array([[-1.5]]), input)[0, 0]


def func_or(input):
    return perceptron(step, np.array([[1, 1]]), np.array([[-0.5]]), input)[0, 0]


def func_nor(input):
    return perceptron(step, np.array([[-1, -1]]), np.array([[0.5]]), input)[0, 0]


def func_nand(input):
    return perceptron(step, np.array([[-1, -1]]), np.array([[1.5]]), input)[0, 0]


def func_xor(input):
    return func_and(np.array([[func_or(input), func_nand(input)]]))


def func_xnor(input):
    return func_or(np.array([[func_nor(input), func_and(input)]]))


def logical_xor(input):
    return func_or(input) & func_nand(input)


def logical_xnor(input):
    return func_nor(input) | func_and(input)


print(logical_xnor(np.array([[0, 0]])))
print(logical_xnor(np.array([[0, 1]])))
print(logical_xnor(np.array([[1, 0]])))
print(logical_xnor(np.array([[1, 1]])))
