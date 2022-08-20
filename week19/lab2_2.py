from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = datasets.load_iris()

# Create our X and y data
X = iris.data[0:100]
y = iris.target[0:100]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

perceptron = Perceptron(max_iter=40, eta0=0.1, random_state=0)
perceptron.fit(X_train, y_train)

print('Accuracy: ', accuracy_score(y_test, perceptron.predict(X_test)))
