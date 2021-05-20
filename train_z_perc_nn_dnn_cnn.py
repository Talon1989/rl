import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import keras


iris = pd.read_csv('data/iris.csv')
mnist = pd.read_csv('data/mnist.csv')

# mnist_v2 = keras.datasets.mnist
# (X_train, y_train), (X_test, y_test) = mnist_v2.load_data()
# first = X_train[0]
# print(first.shape)
# first = first.reshape(-1)
# print(first.shape)


# X = mnist.iloc[:, 1:-1].values / 255.
# y = mnist.iloc[:, 0].values
# first = X[0]
# first = first.reshape([29, 27])


ys = LabelEncoder().fit_transform(iris.iloc[:, -1].values)
X_train, X_test, y_train, y_test = train_test_split(
    iris.iloc[:, 0:-1].values,
    ys,
    train_size=0.5,
    stratify=ys
)


class Perceptron:

    def __init__(self, eta=0.01, seed=1, batch_size=10):
        self.eta = eta
        self.random = np.random.RandomState(seed=seed)
        self.batch_size = batch_size
        self.b, self.w = None, None

    @staticmethod
    def onehot(y):
        Y = np.zeros([np.unique(y).shape[0], y.shape[0]])
        for idx, val in enumerate(y):
            Y[int(val), idx] = 1
        return Y.T

    def net_input(self, X):
        return self.b + np.dot(X, self.w)

    @staticmethod
    def activation(Z):
        return 1 / (1 + np.exp(-np.clip(Z, a_min=-250, a_max=250)))

    def predict(self, X):
        return self.net_input(X)

    def fit(self, X, y, n_iter=1000, l2=0., threshold=1e-26):
        Y = self.onehot(y)
        self.b = np.zeros(Y.shape[1])
        self.w = self.random.normal(0, 0.01, [X.shape[1], Y.shape[1]])
        for i in range(n_iter):
            flag = False
            indices = np.arange(X.shape[0])
            self.random.shuffle(indices)
            for idx in range(0, indices.shape[0] - self.batch_size, self.batch_size):
                batch = indices[idx: idx + self.batch_size]
                error = Y[batch] - self.activation(self.net_input(X[batch]))
                if np.sum(np.abs(error) < threshold):
                    flag = True
                self.b = self.b + self.eta * np.sum(error, axis=0)
                self.w = self.w + self.eta * np.dot(X[batch].T, error) + self.w * l2
            print('epoch: %d' % (i+1))
            if flag:
                break
        return self


perc = Perceptron()
perc.fit(X_train, y_train, l2=0.1)
predictions = np.argmax(perc.predict(X_test), axis=1)
print(
    np.sum(predictions == y_test) / y_test.shape[0]
)














































































































































































































































































































