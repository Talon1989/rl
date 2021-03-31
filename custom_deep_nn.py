import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


np.random.seed(1)


mnist = pd.read_csv('data/mnist.csv').values
y_ = mnist[:, 0]
X_ = mnist[:, 1:] / 255.


class NN:

    def __init__(self, eta=0.01, n_iter=1000, n_hidden=10, batch_size=100, seed=0):
        self.eta = eta
        self.n_iter = n_iter
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.random = np.random.RandomState(seed=seed)
        self.h_b, self.h_W, self.o_b, self.o_W = None, None, None, None

    @staticmethod
    def onehot(y):
        Y = np.zeros(shape=[np.unique(y).shape[0], y.shape[0]])
        for idx, val in enumerate(y):
            Y[int(val), idx] = 1
        return Y.T

    @staticmethod
    def activation(Z):
        return 1 / (1 + np.exp(-np.clip(Z, a_min=-250, a_max=250)))

    def forward(self, X):
        h_Z = self.h_b + np.dot(X, self.h_W)
        h_A = self.activation(h_Z)
        o_Z = self.o_b + np.dot(h_A, self.o_W)
        o_A = self.activation(o_Z)
        return h_Z, h_A, o_Z, o_A

    def predict(self, X):
        _, _, o_Z, _ = self.forward(X)
        return o_Z

    def fit(self, X, y, l2=0.):

        Y = self.onehot(y)
        self.h_b = np.zeros(shape=self.n_hidden)
        self.h_W = self.random.normal(loc=0, scale=0.1, size=[X.shape[1], self.n_hidden])
        self.o_b = np.zeros(shape=Y.shape[1])
        self.o_W = self.random.normal(loc=0, scale=0.1, size=[self.n_hidden, Y.shape[1]])

        for i in range(self.n_iter):
            print('iteration n %d' % (i+1))
            indices = np.arange(X.shape[0])
            self.random.shuffle(indices)

            for idx in range(0, X.shape[0] - self.batch_size, self.batch_size):

                batch = indices[idx: idx+self.batch_size]
                h_Z, h_A, o_Z, o_A = self.forward(X[batch])
                o_delta = Y[batch] - o_A

                self.o_b = self.o_b + self.eta * np.sum(o_delta, axis=0)
                self.o_W = self.o_W + self.eta * np.dot(h_A.T, o_delta) + l2 * self.o_W

                sigmoid_derivative = h_A * (1 - h_A)
                h_delta = np.dot(o_delta, self.o_W.T) * sigmoid_derivative
                self.h_b = self.h_b + self.eta * np.sum(h_delta, axis=0)
                self.h_W = self.h_W + self.eta * np.dot(X[batch].T, h_delta) + l2 * self.h_W

        return self


X_train, X_test, y_train, y_test = train_test_split(
    X_, y_, train_size=0.5, stratify=y_
)

# nn = NN(eta=0.001, n_iter=200, batch_size=10)
# nn.fit(X_train, y_train, l2=0.)
# predictions = nn.predict(X_test)
# print(
#     np.sum(np.argmax(predictions, axis=1) == y_test) / y_test.shape[0]
# )


class DNN:

    def __init__(self, eta=0.01, n_iter=1000, n_hidden_1=10, n_hidden_2=8, batch_size=100, seed=0):
        self.eta = eta
        self.n_iter = n_iter
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.batch_size = batch_size
        self.random = np.random.RandomState(seed=seed)
        self.h1_b, self.h1_W, self.h2_b, self.h2_W, self.o_b, self.o_W = None, None, None, None, None, None

    @staticmethod
    def onehot(y):
        Y = np.zeros(shape=[np.unique(y).shape[0], y.shape[0]])
        for idx, val in enumerate(y):
            Y[int(val), idx] = 1
        return Y.T

    @staticmethod
    def activation(Z):
        return 1 / (1 + np.exp(-np.clip(Z, a_min=-250, a_max=250)))

    def forward(self, X):
        h1_Z = self.h1_b + np.dot(X, self.h1_W)
        h1_A = self.activation(h1_Z)
        h2_Z = self.h2_b + np.dot(h1_A, self.h2_W)
        h2_A = self.activation(h2_Z)
        o_Z = self.o_b + np.dot(h2_A, self.o_W)
        o_A = self.activation(o_Z)
        return h1_Z, h1_A, h2_Z, h2_A, o_Z, o_A

    def predict(self, X):
        _, _, _, _, o_Z, _ = self.forward(X)
        return o_Z

    def fit(self, X, y, l2=0.):

        Y = self.onehot(y)
        self.h1_b = np.zeros(shape=self.n_hidden_1)
        self.h1_W = self.random.normal(loc=0, scale=0.1, size=[X.shape[1], self.n_hidden_1])
        self.h2_b = np.zeros(shape=self.n_hidden_2)
        self.h2_W = self.random.normal(loc=0, scale=0.1, size=[self.n_hidden_1, self.n_hidden_2])
        self.o_b = np.zeros(shape=Y.shape[1])
        self.o_W = self.random.normal(loc=0, scale=0.1, size=[self.n_hidden_2, Y.shape[1]])

        for i in range(self.n_iter):

            print('iteration n %d' % (i+1))
            indices = np.arange(X.shape[0])
            self.random.shuffle(indices)

            for idx in range(0, X.shape[0] - self.batch_size + 1, self.batch_size):

                batch = indices[idx: idx+self.batch_size]
                h1_Z, h1_A, h2_Z, h2_A, o_Z, o_A = self.forward(X[batch])

                o_delta = Y[batch] - o_A
                self.o_b = self.o_b + self.eta * np.sum(o_delta, axis=0)
                self.o_W = self.o_W + self.eta * np.dot(h2_A.T, o_delta) + l2 * self.o_W

                sigmoid_derivative_2 = h2_A * (1 - h2_A)
                h2_delta = np.dot(o_delta, self.o_W.T) * sigmoid_derivative_2
                self.h2_b = self.h2_b + self.eta * np.sum(h2_delta, axis=0)
                self.h2_W = self.h2_W + self.eta * np.dot(h1_A.T, h2_delta) + l2 * self.h2_W

                sigmoid_derivative_1 = h1_A * (1 - h1_A)
                h1_delta = np.dot(h2_delta, self.h2_W.T) * sigmoid_derivative_1
                self.h1_b = self.h1_b + self.eta * np.sum(h1_delta, axis=0)
                self.h1_W = self.h1_W + self.eta * np.dot(X[batch].T, h1_delta) + l2 * self.h1_W

        return self


dnn = DNN(eta=0.001, n_iter=1000, batch_size=20)
dnn.fit(X_train, y_train, l2=0.0)
predictions = dnn.predict(X_test)
print(
    np.sum(np.argmax(predictions, axis=1) == y_test) / y_test.shape[0]
)
# predictions = dnn.predict(X_train)
# print(
#     np.sum(np.argmax(predictions, axis=1) == y_train) / y_train.shape[0]
# )



















































































































































































































































































































































































