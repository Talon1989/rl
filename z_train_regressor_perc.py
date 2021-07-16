import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


iris = pd.read_csv('data/iris.csv')
X_ = iris.iloc[:, 0:-2].values
y_ = iris.iloc[:, -2].values


X_train, X_test, y_train, y_test = train_test_split(
    X_, y_, train_size=0.5
)


def projection_weights(X, y):
    X = np.hstack([
        np.ones([X.shape[0], 1]), X
    ])
    w = np.dot(
        np.linalg.inv(np.dot(X.T, X)),
        np.dot(X.T, y)
    )
    return w


class RegressorPerc:

    def __init__(self, n_iter, eta, batch_size=10):
        assert n_iter > 0
        assert 0 <= eta <= 1
        self.n_iter = n_iter
        self.eta = eta
        self.batch_size = batch_size
        self.b, self.w = None, None

    def net_input(self, X):
        return self.b + np.dot(X, self.w)

    @staticmethod
    def relu(Z):
        Z[Z < 0] = 0
        return Z

    def predict(self, X):
        return self.net_input(X)

    def fit(self, X, y):
        assert X.shape[0] == y.shape[0]
        self.b = 0
        self.w = np.zeros(X.shape[1])
        for i in range(self.n_iter):
            indices = np.arange(0, X.shape[0])
            np.random.shuffle(indices)
            iter_error = 0
            for idx in range(0, X.shape[0] - self.batch_size, self.batch_size):
                batch = indices[idx: idx+self.batch_size]
                error = y[batch] - self.relu(self.net_input(X[batch]))
                iter_error += np.sum(error ** 2)
                self.b = self.b + self.eta * np.sum(error)
                self.w = self.w + self.eta * np.dot(X[batch].T, error)
            print('Iteration %d completed, loss: %f' % (i, iter_error))
        return self


perc = RegressorPerc(500, 0.0001)
perc.fit(X_train, y_train)


proj_weights = projection_weights(X_train, y_train)
perc_weights = np.hstack([perc.b, perc.w])


proj_prediction = proj_weights[0] + np.dot(X_test, proj_weights[1:])
perc_prediction = perc.predict(X_test)


print('projection prediction score: %f' % r2_score(y_test, proj_prediction))
print('regression prediction score: %f' % r2_score(y_test, perc_prediction))











































































































































































































