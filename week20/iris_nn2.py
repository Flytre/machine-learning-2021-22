import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

ova = OneVsRestClassifier(SVC()).fit(X_train, y_train)
print(ova.score(X_test, y_test))
