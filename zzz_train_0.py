import random
import gym
import numpy as np
import pandas as pd
from collections import deque
import tensorflow
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import r2_score


def r_2(y_true: np.array, y_pred: np.array):
    mean_y = np.sum(y_true) / y_true.shape[0]
    ss_tot = np.sum((y_true - mean_y) ** 2)
    try:
        y_pred.shape[1]
        ss_res = 0
        for i in range(y_pred.shape[0]):
            ss_res += (y_true[i] - y_pred[i][0]) ** 2
    except IndexError:
        ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_tot)


class NN:

    def __init__(self, n_epochs=500, eta=1/1000, n_hidden=10, batch_size=2**5, classification=True):
        self.n_epochs = n_epochs
        self.eta = eta
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.classification = classification
        self.b_h, self.W_h, self.b_o, self.W_o = None, None, None, None

    @staticmethod
    def onehot(y):
        Y = np.zeros([np.unique(y).shape[0], y.shape[0]])
        for idx, val in enumerate(y):
            Y[int(val), idx] = 1
        return Y.T

    @staticmethod
    def activation_classification(X):
        return 1 / (1 + np.exp(-np.clip(X, a_min=-250, a_max=250)))

    @staticmethod
    def activation_regression(Z):
        Z[Z < 0] = 0
        return Z

    def forward(self, X):
        Z_h = self.b_h + np.dot(X, self.W_h)
        if self.classification:
            A_h = self.activation_classification(Z_h)
        else:
            A_h = self.activation_regression(Z_h)
        Z_o = self.b_o + np.dot(A_h, self.W_o)
        if self.classification:
            A_o = self.activation_classification(Z_o)
        else:
            A_o = self.activation_regression(Z_o)
        return Z_h, A_h, Z_o, A_o

    def predict(self, X):
        _, _, Z_o, _ = self.forward(X)
        return Z_o

    def fit(self, X, y):
        if self.classification:
            self.fit_classification(X, y)
        else:
            self.fit_regression(X, y)

    def fit_classification(self, X, y):
        Y = self.onehot(y)
        self.b_h = np.zeros(self.n_hidden)
        self.W_h = np.zeros([X.shape[1], self.n_hidden])
        self.b_o = np.zeros(Y.shape[1])
        self.W_o = np.zeros([self.n_hidden, Y.shape[1]])
        for i in range(self.n_epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            for idx in range(0, X.shape[0] - self.batch_size, self.batch_size):
                batch = indices[idx: idx + self.batch_size]
                Z_h, A_h, Z_o, A_o = self.forward(X[batch])
                delta_o = Y[batch] - A_o
                delta_h = np.dot(delta_o, self.W_o.T) * A_h * (1 - A_h)
                self.b_o = self.b_o + self.eta * np.sum(delta_o, axis=0)
                self.W_o = self.W_o + self.eta * np.dot(A_h.T, delta_o)
                self.b_h = self.b_h + self.eta * np.sum(delta_h, axis=0)
                self.W_h = self.W_h + self.eta * np.dot(X[batch].T, delta_h)
        return self

    @staticmethod
    def regression_derivative(Z):
        Z[Z < 0] = 0
        Z[Z > 0] = 1
        return Z

    def fit_regression(self, X, y):
        self.b_h = np.zeros(self.n_hidden)
        self.W_h = np.random.normal(loc=0, scale=0.01, size=[X.shape[1], self.n_hidden])
        self.b_o = 0
        self.W_o = np.random.normal(loc=0, scale=0.01, size=[self.n_hidden, 1])
        for i in range(self.n_epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            for idx in range(0, X.shape[0] - self.batch_size, self.batch_size):
                batch = indices[idx: idx + self.batch_size]
                Z_h, A_h, Z_o, A_o = self.forward(X[batch])
                delta_o = y[batch].reshape(-1, 1) - A_o
                delta_h = np.dot(delta_o, self.W_o.T) * self.regression_derivative(A_h)
                self.b_o = self.b_o + self.eta * np.sum(delta_o, axis=0)
                self.W_o = self.W_o + self.eta * np.dot(A_h.T, delta_o)
                self.b_h = self.b_h + self.eta * np.sum(delta_h, axis=0)
                self.W_h = self.W_h + self.eta * np.dot(X[batch].T, delta_h)
        return self


# iris = pd.read_csv('data/iris.csv')
# X_ = iris.iloc[:, 0:-1].values
# y_ = LabelEncoder().fit_transform(iris.iloc[:, -1].values)
# X_train, X_test, y_train, y_test = train_test_split(X_, y_, stratify=y_, test_size=1/2)
# nn = NN(5_000)
# nn.fit(X_train, y_train)
# predictions = np.argmax(nn.predict(X_test), axis=1)
# print(
#     np.sum(predictions == y_test) / y_test.shape[0]
# )

iris = pd.read_csv('data/iris.csv')
X_ = iris.iloc[:, 0:-2].values
y_ = iris.iloc[:, -2].values
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=1/2)
nn = NN(n_epochs=4_000, eta=0.0001, classification=False)
nn.fit(X_train, y_train)
predictions = nn.predict(X_test)
print('np r2 score: %.4f' % r2_score(y_test, predictions))
print('custom r2 score: %.4f' % r_2(y_test, predictions))


def q_learning():
    # target = np.array([0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0])
    n_episodes = 5_000
    alpha = 2/10
    gamma = 99/100
    epsilon_decay = 999/1000
    env = gym.make('FrozenLake-v0')
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q = np.zeros([n_states, n_actions])
    epsilon = 1
    def e_greedy(s):
        if np.random.uniform(0, 1) < epsilon or np.sum(np.abs(s)) == 0:
            return np.random.randint(0, n_actions)
        else:
            return np.argmax(q[s])
    for e in range(n_episodes):
        s = env.reset()
        while True:
            a = e_greedy(s)
            s_, r, done, _ = env.step(a)
            q[s, a] = q[s, a] + alpha * (r + gamma * np.max(q[s_]) - q[s, a])
            epsilon = epsilon * epsilon_decay
            if done:
                break
            s = s_
    return q


# pol = q_learning()


class DQL:

    def __init__(self, n_episodes=3_000, gamma=95/100, batch_size=2**5):
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = gym.make('CartPole-v1')  # v0 = max reward 200 ; v1 = max reward 500
        self.state_size = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.epsilon = 1
        self.epsilon_decay = 999/1000
        self.epsilon_min = 1/1000
        self.replay_buffer = deque(maxlen=100)
        self.main_nn = self.build_nn()
        self.target_nn = self.build_nn()

    def build_nn(self):
        model = keras.Sequential([
            keras.layers.Dense(2**5, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(2**5, activation='relu'),
            keras.layers.Dense(self.n_actions, activation='linear')
        ])
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        return model

    def store_transition(self, s, a, r, s_, done):
        self.replay_buffer.append([s, a, r, s_, done])

    def update_target_nn(self):
        self.target_nn.set_weights(self.main_nn.get_weights())

    def e_greedy(self, s):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            return np.argmax(self.main_nn.predict(s))

    def replay(self):
        minibatch = np.array(random.sample(self.replay_buffer, self.batch_size))
        sstates = minibatch[:, 0]
        actions = minibatch[:, 1].astype(np.int)
        rewards = minibatch[:, 2].astype(np.float)
        sstates_ = minibatch[:, 3]
        dones = minibatch[:, 4].astype(np.bool)
        states, states_ = [], []
        for i in range(sstates.shape[0]):
            states.append(sstates[i][0])
            states_.append(sstates_[i][0])
        states = np.array(states)
        states_ = np.array(states_)
        targets = rewards + self.gamma * np.max(self.target_nn.predict(states_), axis=1) * (1 - dones)
        q_values = self.main_nn.predict(states)
        for i in range(q_values.shape[0]):
            q_values[actions[i]] = targets[i]
        self.main_nn.fit(states, q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def replay_test(self):
        minibatch = np.array(random.sample(self.replay_buffer, self.batch_size))
        states = minibatch[:, 0]
        actions = minibatch[:, 1].astype(np.int)
        rewards = minibatch[:, 2].astype(np.float)
        states_ = minibatch[:, 3]
        dones = minibatch[:, 4].astype(np.bool)
        targets = rewards + self.gamma * np.max(self.target_nn.predict(states_), axis=1) * (1 - dones)
        q_values = self.main_nn.predict(states)
        for i in range(q_values.shape[0]):
            q_values[actions[i]] = targets[i]
        self.main_nn.fit(states, q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def train(self):
        for e in range(self.n_episodes):
            s = self.env.reset()
            s = np.reshape(s, [1, self.state_size])
            score = 0
            while True:
                self.env.render()
                a = self.e_greedy(s)
                s_, r, done, _ = self.env.step(a)
                s_ = np.reshape(s_, [1, self.state_size])
                r = r if not done else -100
                self.store_transition(s, a, r, s_, done)
                if done:
                    print('Episode: %d , epsilon: %.3f , score %d' % (e, self.epsilon, score))
                    break
                s = s_
                score += 1
            if len(self.replay_buffer) == 100:
                self.replay()
            if e > 0 and e % 10 == 0:
                print('Updating target network')
                self.update_target_nn()


# dql = DQL()
# dql.train()













































































































































































































































































































































































































































































