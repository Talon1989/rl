import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow as tf


iris = pd.read_csv('data/iris.csv')
mnist = pd.read_csv('data/mnist.csv')


X = mnist.iloc[:, 1:-1].values / 255.
y = mnist.iloc[:, 0].values
first = X[0]
first = first.reshape([29, 27])
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.5,
    stratify=y
)


# ys = LabelEncoder().fit_transform(iris.iloc[:, -1].values)
# X_train, X_test, y_train, y_test = train_test_split(
#     iris.iloc[:, 0:-1].values,
#     ys,
#     train_size=0.5,
#     stratify=ys
# )


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


# perc = Perceptron()
# perc.fit(X_train, y_train, l2=0.1)
# predictions = np.argmax(perc.predict(X_test), axis=1)
# print(
#     np.sum(predictions == y_test) / y_test.shape[0]
# )


class test_DNN:

    def __init__(self, eta=0.01, seed=1, batch_size=10, n_h1=10, n_h2=5):
        self.eta = eta
        self.random = np.random.RandomState(seed=seed)
        self.batch_size = batch_size
        self.b_h1, self.W_h1 = None, None
        self.b_h2, self.W_h2 = None, None
        self.b_o, self.W_o = None, None
        self.n_h1, self.n_h2 = n_h1, n_h2

    @staticmethod
    def onehot(y):
        Y = np.zeros([np.unique(y).shape[0], y.shape[0]])
        for idx, val in enumerate(y):
            Y[int(val), idx] = 1
        return Y.T

    @staticmethod
    def activation(Z):
        return 1 / (1 + np.exp(-Z))

    def forward(self, X):
        Z_1 = self.b_h1 + np.dot(X, self.W_h1)
        A_1 = self.activation(Z_1)
        Z_2 = self.b_h2 + np.dot(A_1, self.W_h2)
        A_2 = self.activation(Z_2)
        Z_o = self.b_o + np.dot(A_2, self.W_o)
        A_o = self.activation(Z_o)
        return Z_1, A_1, Z_2, A_2, Z_o, A_o

    def predict(self, X):
        _, _, _, _, Z_o, _ = self.forward(X)
        return Z_o

    def fit(self, X, y, n_iter=500, l2=0):
        Y = self.onehot(y)
        self.b_h1 = np.zeros(self.n_h1)
        # self.W_h1 = self.random.normal(0, 0.01, size=[X.shape[1], self.n_h1])
        self.W_h1 = np.zeros([X.shape[1], self.n_h1])
        self.b_h2 = np.zeros(self.n_h2)
        # self.W_h2 = self.random.normal(0, 0.01, size=[self.n_h1, self.n_h2])
        self.W_h2 = np.zeros([self.n_h1, self.n_h2])
        self.b_o = np.zeros(Y.shape[1])
        # self.W_o = self.random.normal(0, 0.01, size=[self.n_h2, Y.shape[1]])
        self.W_o = np.zeros([self.n_h2, Y.shape[1]])
        error = []
        bb = []
        print('starting')
        for i in range(n_iter):
            bb.append(self.b_h2)
            indices = np.arange(X.shape[0])
            self.random.shuffle(indices)
            epoch_error = 0
            for idx in range(0, X.shape[0] - self.batch_size, self.batch_size):
                batch = indices[idx: idx + self.batch_size]
                Z_1, A_1, Z_2, A_2, Z_o, A_o = self.forward(X[batch])

                delta_o = Y[batch] - A_o
                delta_h2 = np.dot(delta_o, self.W_o.T) * (A_2 * (1 - A_2))
                # if i == 0:
                #     print(idx)
                #     print(delta_o)
                delta_h1 = np.dot(delta_h2, self.W_h2.T) * (A_1 * (1 - A_1))

                self.b_o = self.b_o + self.eta * np.sum(delta_o, axis=0)
                self.W_o = self.W_o + self.eta * np.dot(A_2.T, delta_o)
                self.b_h2 = self.b_h2 + self.eta * np.sum(delta_h2, axis=0)
                self.W_h2 = self.W_h2 + self.eta * np.dot(A_1.T, delta_h2)
                self.b_h1 = self.b_h1 + self.eta * np.sum(delta_h1, axis=0)
                self.W_h1 = self.W_h1 + self.eta * np.dot(X[batch].T, delta_h1)

                epoch_error += np.sum(np.abs(delta_o))

            error.append(epoch_error)

            # print('epoch %d' % (i+1))
            if i >= 1000:
                break

        return error, bb


class NN:

    def __init__(self, eta=0.01, seed=1, batch_size=10):
        self.eta = eta
        self.random = np.random.RandomState(seed=seed)
        self.batch_size = batch_size
        self.b_h, self.W_h = None, None
        self.b_o, self.W_o = None, None

    @staticmethod
    def onehot(y):
        Y = np.zeros([np.unique(y).shape[0], y.shape[0]])
        for idx, val in enumerate(y):
            Y[int(val), idx] = 1
        return Y.T

    @staticmethod
    def activation(Z):
        return 1 / (1 + np.exp(-Z))

    def forward(self, X):
        Z_h = self.b_h + np.dot(X, self.W_h)
        A_h = self.activation(Z_h)
        Z_o = self.b_o + np.dot(A_h, self.W_o)
        A_o = self.activation(Z_o)
        return Z_h, A_h, Z_o, A_o

    def predict(self, X):
        _, _, Z_o, _ = self.forward(X)
        return Z_o

    def fit(self, X, y, n_iter=1000, n_hidden=10):
        Y = self.onehot(y)
        self.b_h = np.zeros(n_hidden)
        self.W_h = self.random.normal(0, 0.01, [X.shape[1], n_hidden])
        self.b_o = np.zeros(Y.shape[1])
        self.W_o = self.random.normal(0, 0.01, [n_hidden, Y.shape[1]])
        error = []
        for i in range(n_iter):
            print('epoch %d' % (i+1))
            indices = np.arange(X.shape[0])
            self.random.shuffle(indices)
            epoch_error = 0
            for idx in range(0, X.shape[0] - self.batch_size, self.batch_size):
                batch = indices[idx: idx + self.batch_size]
                Z_h, A_h, Z_o, A_o = self.forward(X[batch])

                delta_o = - (Y[batch] - A_o)
                delta_h = np.dot(delta_o, self.W_o.T) * (A_h * (1 - A_h))

                self.b_o = self.b_o - self.eta * np.sum(delta_o, axis=0)
                self.W_o = self.W_o - self.eta * np.dot(A_h.T, delta_o)
                self.b_h = self.b_h - self.eta * np.sum(delta_h, axis=0)
                self.W_h = self.W_h - self.eta * np.dot(X[batch].T, delta_h)

                epoch_error += np.sum(np.abs(delta_o))

            error.append(epoch_error)

        return error


# nn = test_DNN(eta=0.001, batch_size=10)

# nn = NN(eta=0.001, batch_size=40)
# er = nn.fit(X_train, y_train, n_iter=1000)
#
# print(y_test[20:50])
# print(
#     np.argmax(nn.predict(X_test[20:50]), axis=1)
# )
# print()
# print(
#     np.sum(np.argmax(nn.predict(X_test), axis=1) == y_test) / y_test.shape[0]
# )
#
# plt.plot(er)
# plt.show()
# plt.clf()
#
#

mnist_v2 = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist_v2.load_data()
y_train = NN().onehot(y_train)
y_test = NN().onehot(y_test)
X_train = X_train[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]

cnn = keras.Sequential([
    keras.layers.Conv2D(
        filters=32, kernel_size=[5, 5], strides=[1, 1],
        padding='same', data_format='channels_last', activation='relu'
    ),
    keras.layers.MaxPool2D([2, 2]),
    keras.layers.Conv2D(
        filters=64, kernel_size=[5, 5], strides=[1, 1],
        padding='same', activation='relu'
    ),
    keras.layers.MaxPool2D([2, 2]),
    keras.layers.Flatten(),
    keras.layers.Dense(units=1024, activation='relu'),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(units=10, activation='softmax')
])
cnn.build(input_shape=(None, 28, 28, 1))
cnn.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)
cnn.fit(X_train, y_train, batch_size=10, epochs=2)


print('predicting first train data')
print(
    np.argmax(
        cnn.predict(X_train[0:1])[0]
    )
)






























































































































































































































































































