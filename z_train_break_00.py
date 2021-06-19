import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import gym


iris = pd.read_csv('data/iris.csv')
mnist = pd.read_csv('data/mnist.csv')


# X_ = iris.iloc[:, :-1].values
# y_ = LabelEncoder().fit_transform(iris.iloc[:, -1].values)
# X_train, X_test, y_train, y_test = train_test_split(
#     X_, y_, train_size=0.6, stratify=y_
# )


class Perceptron:

    def __init__(self, seed=1):
        self.b, self.W = None, None
        self.random = np.random.RandomState(seed=seed)

    @staticmethod
    def onehot(y):
        Y = np.zeros([np.unique(y).shape[0], y.shape[0]])
        for idx, val in enumerate(y):
            Y[int(val), idx] = 1
        return Y.T

    def net_input(self, X):
        return self.b + np.dot(X, self.W)

    @staticmethod
    def activation(Z):
        return 1 / (1 + np.exp(-np.clip(Z, a_min=-250, a_max=250)))

    def predict(self, X):
        return self.net_input(X)

    def fit(self, X, y, n_iter=10_000, batch_size=10, eta=0.01, l2=0):
        Y = self.onehot(y)
        self.b = np.zeros(Y.shape[1])
        self.W = self.random.normal(0, 0.01, size=[X.shape[1], Y.shape[1]])
        for i in range(n_iter):
            indices = np.arange(0, X.shape[0])
            np.random.shuffle(indices)
            for idx in range(0, X.shape[0] - batch_size, batch_size):
                batch = indices[idx: idx + batch_size]
                error = Y[batch] - self.activation(self.net_input(X[batch]))
                self.b = self.b + eta * np.sum(error, axis=0)
                self.W = self.W + eta * np.dot(X[batch].T, error) + l2 * self.W
        return self


# perc = Perceptron()
# perc.fit(X_train, y_train)
# predictions = perc.predict(X_test)
# print(
#     np.sum(np.argmax(predictions, axis=1) == y_test) / y_test.shape[0]
# )


class NN:

    def __init__(self, seed=1, n_hidden=10):
        self.b_h, self.W_h = None, None
        self.b_o, self.W_o = None, None
        self.n_hidden = n_hidden
        self.random = np.random.RandomState(seed=seed)

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

    def fit(self, X, y, n_iter=5_000, batch_size=10, eta=0.01, l2=0):
        Y = self.onehot(y)
        self.b_h = np.zeros(self.n_hidden)
        self.W_h = self.random.normal(0, 0.01, size=[X.shape[1], self.n_hidden])
        self.b_o = np.zeros(Y.shape[1])
        self.W_o = self.random.normal(0, 0.01, size=[self.n_hidden, Y.shape[1]])
        for i in range(n_iter):
            indices = np.arange(0, X.shape[0])
            np.random.shuffle(indices)
            for idx in range(0, X.shape[0] - batch_size, batch_size):
                batch = indices[idx: idx+batch_size]
                Z_h, A_h, Z_o, A_o = self.forward(X[batch])
                delta_o = - (Y[batch] - A_o)
                delta_h = np.dot(delta_o, self.W_o.T) * (A_h - A_h ** 2)
                self.b_o = self.b_o - eta * np.sum(delta_o, axis=0)
                self.W_o = self.W_o - eta * np.dot(A_h.T, delta_o) + l2 * self.W_o
                self.b_h = self.b_h - eta * np.sum(delta_h, axis=0)
                self.W_h = self.W_h - eta * np.dot(X[batch].T, delta_h) + l2 * self.W_h
        return self


# nn = NN()
# nn.fit(X_train, y_train)
# predictions = nn.predict(X_test)
# print(
#     np.sum(np.argmax(predictions, axis=1) == y_test) / y_test.shape[0]
# )


# (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
# X_train, X_test = X_train / 253., X_test / 253.
# xx_train = np.array(
#     [X_train[i].reshape([1, -1]) for i in range(X_train.shape[0])]
# )
# X_train, X_test = X_train[..., tf.newaxis], X_test[..., tf.newaxis]
# y_train, y_test = NN().onehot(y_train), NN().onehot(y_test)

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
    keras.layers.Dense(
        units=1024, activation='relu'
    ),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(
        units=10, activation='softmax'
    )
])
cnn.build(input_shape=(None, 28, 28, 1))
cnn.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)
# cnn.fit(X_train, y_train, batch_size=10, epochs=2)


###############################################################################


# p_s = np.array([0.2, 0.4, 0.7])
# banners = np.ones([3, 2])
# for i in range(1000):
#     pop_up = np.argmax(
#         np.random.beta(banners[:, 0], banners[:, 1])
#     )
#     R = np.random.binomial(1, p_s[pop_up])
#     banners[pop_up][0] += R
#     banners[pop_up][1] += 1 - R
# print(banners)


target = np.array([0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0])
env = gym.make('FrozenLake-v0')
n_states = env.observation_space.n
n_actions = env.action_space.n
P = env.P


def value_iteration(threshold=1e-16):
    gamma = 0.99
    v = np.zeros(n_states)
    while True:
        old_v = v.copy()
        for s in range(n_states):
            v[s] = np.max([
                np.sum([
                    (r + gamma * old_v[s_]) * p
                    for p, s_, r, _ in P[s][a]
                ]) for a in range(n_actions)
            ])
        if np.sum(np.abs(old_v - v)) < threshold:
            break
    policy = np.zeros(n_states)
    for s in range(n_states):
        policy[s] = np.argmax([
            np.sum([
                (r + gamma * v[s_]) * p
                for p, s_, r, _ in P[s][a]
            ]) for a in range(n_actions)
        ])
    return policy, v


# pol, _ = value_iteration()


def mc(n_iter=10_000):

    from collections import defaultdict
    env = gym.make('Blackjack-v0')
    n_actions = env.action_space.n
    q = defaultdict(float)
    total_rewards = defaultdict(float)
    n = defaultdict(int)

    def e_greedy(s, epsilon=0.1):
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, n_actions)
        else:
            action = np.argmax([
                q[(s, a)] for a in range(n_actions)
            ])
        return action

    def play_episode():
        episode = []
        s = env.reset()
        while True:
            a = e_greedy(s)
            s_, r, done, _ = env.step(a)
            episode.append([s, a, r])
            if done:
                break
            s = s_
        return episode

    for i in range(n_iter):
        episode = play_episode()
        s_a_pairs = [(s, a) for s, a, _ in episode]
        rewards = [r for _, _, r in episode]
        for t, (s, a, _) in enumerate(episode):
            if (s, a) not in s_a_pairs[:t]:
                total_rewards[(s, a)] += np.sum(rewards[t:])
                n[(s, a)] += 1
                q[(s, a)] = total_rewards[(s, a)] / n[(s, a)]

    return pd.merge(
        pd.DataFrame(q.items(), columns=['(s, a)', 'value']),
        pd.DataFrame(n.items(), columns=['(s, a)', 'n']),
        on='(s, a)'
    )


def q_learning(n_iter=10_000, gamma=0.99, alpha=0.2):

    q = np.zeros([n_states, n_actions])

    def e_greedy(s, epsilon=0.1):
        if np.random.uniform(0, 1) < epsilon or np.sum(np.abs(q[s])) == 0:
            action = np.random.randint(0, n_actions)
        else:
            action = np.argmax(q[s])
        return action

    for i in range(n_iter):
        s = env.reset()
        while True:
            a = e_greedy(s)
            s_, r, done, _ = env.step(a)
            q[s, a] = q[s, a] + alpha * (r + gamma * q[s_, np.argmax(q[s_])] - q[s, a])
            if done:
                break
            s = s_

    return q






















































































































































































































