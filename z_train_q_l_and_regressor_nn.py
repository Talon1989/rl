import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import gym


def q_learning(n_iter=10_000, alpha=0.2, gamma=0.99):

    env = gym.make('FrozenLake-v0')
    n_actions = env.action_space.n  # l, d, r ,u
    n_states = env.observation_space.n
    target = np.array([0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0])
    q = np.zeros([n_states, n_actions])

    def e_greedy(s, e=0.1):
        if np.random.uniform(0, 1) < e or np.sum(np.abs(q[s])) == 0:
            return np.random.randint(0, n_actions)
        return np.argmax(q[s])

    for i in range(n_iter):
        print(i)
        s = env.reset()
        while True:
            a = e_greedy(s)
            s_, r, done, _ = env.step(a)
            q[s, a] = q[s, a] + alpha * (r + gamma * q[s_, np.argmax(q[s_])] - q[s, a])
            if done:
                break
            s = s_

    return q


class ClassifierNN:

    def __init__(self, eta=0.01, n_iter=5_000, n_hidden=10, batch_size=10):
        self.eta = eta
        self.n_iter = n_iter
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.b_h, self.W_h, self.b_o, self.W_o = None, None, None, None

    @staticmethod
    def onehot(y):
        Y = np.zeros([np.unique(y).shape[0], y.shape[0]])
        for idx, val in enumerate(y):
            Y[int(val), idx] = 1
        return Y.T

    @staticmethod
    def activation(Z):
        return 1 / (1 + np.exp(-np.clip(Z, a_min=-250, a_max=250)))

    def forward(self, X):
        Z_h = self.b_h + np.dot(X, self.W_h)
        A_h = self.activation(Z_h)
        Z_o = self.b_o + np.dot(A_h, self.W_o)
        A_o = self.activation(Z_o)
        return Z_h, A_h, Z_o, A_o

    def predict(self, X):
        _, _, Z_o, _ = self.forward(X)
        return Z_o

    def fit(self, X, y, l2=0):
        Y = self.onehot(y)
        self.b_h = np.zeros(self.n_hidden)
        self.W_h = np.random.normal(loc=0, scale=0.01, size=[X.shape[1], self.n_hidden])
        self.b_o = np.zeros(Y.shape[1])
        self.W_o = np.random.normal(loc=0, scale=0.01, size=[self.n_hidden, Y.shape[1]])
        for i in range(self.n_iter):
            indices = np.arange(0, X.shape[0])
            np.random.shuffle(indices)
            error = 0
            for idx in range(0, X.shape[0] - self.batch_size, self.batch_size):
                batch = indices[idx: idx + self.batch_size]
                Z_h, A_h, Z_o, A_o = self.forward(X[batch])
                delta_o = Y[batch] - A_o
                delta_h = np.dot(delta_o, self.W_o.T) * A_h * (1 - A_h)
                self.b_o = self.b_o + self.eta * np.sum(delta_o, axis=0)
                self.W_o = self.W_o + self.eta * np.dot(A_h.T, delta_o) + l2 * self.W_o
                self.b_h = self.b_h + self.eta * np.sum(delta_h, axis=0)
                self.W_h = self.W_h + self.eta * np.dot(X[batch].T, delta_h) + l2 * self.W_h
                error += np.sum(delta_o ** 2)
            print('Epoch: %d, loss: %f' % (i, error))
        return self


class RegressorNN:

    def __init__(self, eta=0.01, n_iter=5_000, n_hidden=10, batch_size=10):
        self.eta = eta
        self.n_iter = n_iter
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.b_h, self.W_h, self.b_o, self.W_o = None, None, None, None

    @staticmethod
    def activation(Z):
        Z[Z < 0] = 0
        return Z

    def forward(self, X):
        Z_h = self.b_h + np.dot(X, self.W_h)
        A_h = self.activation(Z_h)
        Z_o = self.b_o + np.dot(A_h, self.W_o)
        A_o = self.activation(Z_o)
        return Z_h, A_h, Z_o, A_o

    @staticmethod
    def relu_derivative(Z):
        Z[Z < 0] = 0
        Z[Z > 0] = 1
        return Z

    def predict(self, X):
        _, _, Z_o, _ = self.forward(X)
        return Z_o

    def fit(self, X, y, l2=0):
        self.b_h = np.zeros(self.n_hidden)
        self.W_h = np.random.normal(loc=0, scale=0.01, size=[X.shape[1], self.n_hidden])
        self.b_o = 0
        self.W_o = np.random.normal(loc=0, scale=0.01, size=[self.n_hidden, 1])
        for i in range(self.n_iter):
            indices = np.arange(0, X.shape[0])
            np.random.shuffle(indices)
            error = 0
            for idx in range(0, X.shape[0] - self.batch_size, self.batch_size):
                batch = indices[idx: idx + self.batch_size]
                Z_h, A_h, Z_o, A_o = self.forward(X[batch])
                delta_o = y[batch].reshape(-1, 1) - A_o
                delta_h = np.dot(delta_o, self.W_o.T) * self.relu_derivative(A_h)
                self.b_o = self.b_o + self.eta * np.sum(delta_o, axis=0)
                self.W_o = self.W_o + self.eta * np.dot(A_h.T, delta_o) + l2 * self.W_o
                self.b_h = self.b_h + self.eta * np.sum(delta_h, axis=0)
                self.W_h = self.W_h + self.eta * np.dot(X[batch].T, delta_h) + l2 * self.W_h
                error += np.sum(delta_o ** 2)
            print('Epoch: %d, loss: %f' % (i, error))
        return self


iris = pd.read_csv('data/iris.csv')
# X_ = iris.iloc[:, 0:-1].values
# y_ = LabelEncoder().fit_transform(iris.iloc[:, -1].values)
# X_train, X_test, y_train, y_test = train_test_split(
#     X_, y_, train_size=0.5, stratify=y_
# )
#
#
# nn = ClassifierNN(n_iter=1_000)
# nn.fit(X_train, y_train)
# print()
# predictions = np.argmax(nn.predict(X_test), axis=1)
# print(
#     np.sum(predictions == y_test) / y_test.shape[0]
# )


X_ = iris.iloc[:, 0:-2].values
y_ = iris.iloc[:, -2].values
X_train, X_test, y_train, y_test = train_test_split(
    X_, y_, train_size=0.5
)
nn = RegressorNN(n_iter=5_000, eta=0.001, n_hidden=5)
nn.fit(X_train, y_train)
predictions = nn.predict(X_test)
print(r2_score(y_test, predictions))







































































































































































































































































