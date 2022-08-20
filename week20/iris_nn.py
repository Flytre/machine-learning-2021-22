import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# sources used:
# https://annisap.medium.com/build-your-first-neural-network-in-python-c80c1afa464
# https://www.kaggle.com/azzion/iris-data-set-classification-using-neural-network


df = pd.read_csv('iris.csv')

df["class"] = df["class"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}).astype(int)

X = df[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']]
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

unscaled_features = X_train
sc = StandardScaler()

X_train_array = sc.fit_transform(X_train.values)
X_train = pd.DataFrame(X_train_array, index=X_train.index, columns=X_train.columns)
X_test_array = sc.transform(X_test.values)
X_test = pd.DataFrame(X_test_array, index=X_test.index, columns=X_test.columns)

LEARNING_RATE_INIT = 1

mlp = MLPClassifier(hidden_layer_sizes=(10), solver='sgd', learning_rate_init=LEARNING_RATE_INIT, max_iter=500)
# mlp.fit(X_train, y_train)
#
# print(mlp.score(X_test, y_test))


epoch_x = list()
loss_y = list()
coef_y = list()
mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))
for i in range(0, 250):
    mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))
    epoch_x.append(i)
    coef_y.append(mlp.coefs_[0][2][1])
    loss_y.append(mlp.loss_)


plt.plot(epoch_x, loss_y)
plt.title("LOSS: learning rate = " + str(LEARNING_RATE_INIT))
plt.show()

plt.plot(epoch_x, coef_y)
plt.title("WEIGHT: learning rate = " + str(LEARNING_RATE_INIT))
plt.show()
