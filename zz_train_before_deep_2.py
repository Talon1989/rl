import numpy as np
import gym
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


env = gym.make('FrozenLake-v0')
n_actions = env.action_space.n  # l, d, r ,u
n_states = env.observation_space.n
target = np.array([0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0])


def pol_iter(threshold=1e-16, gamma=0.99):
    P = env.P
    pol = np.zeros(n_states)
    while True:
        v = np.zeros(n_states)
        while True:
            old_v = v.copy()
            for s in range(n_states):
                v[s] = np.sum([
                    (r + gamma * old_v[s_]) * p
                    for p, s_, r, _ in P[s][pol[s]]
                ])
            if np.sum(np.abs(old_v - v)) < threshold:
                break
        old_pol = pol.copy()
        for s in range(n_states):
            pol[s] = np.argmax([
                np.sum([
                    (r + gamma * v[s_]) * p
                    for p, s_, r, _ in P[s][a]
                ])
                for a in range(n_actions)
            ])
        if (old_pol == pol).all():
            break
    return pol


def mc(n_iter=10_000):

    from collections import defaultdict
    env = gym.make('Blackjack-v0')
    n_actions = env.action_space.n
    q = defaultdict(float)
    total_rewards = defaultdict(float)
    n = defaultdict(int)

    def e_greedy(s, e=0.1):
        if np.random.uniform(0, 1) < e:
            return np.random.randint(0, n_actions)
        else:
            return np.argmax([
                q[s, a] for a in range(n_actions)
            ])

    def play_episode():
        episode = []
        s = env.reset()
        while True:
            action = e_greedy(s)
            s_, r, done, _ = env.step(action)
            episode.append([s, action, r])
            if done:
                break
        return episode

    for _ in range(n_iter):
        episode = play_episode()
        s_a_pairs = [(s, a) for s, a, _ in episode]
        rewards = [r for _, _, r in episode]
        for t, (s, a, _) in enumerate(episode):
            if (s, a) not in s_a_pairs[:t]:
                total_rewards[s, a] += np.sum(rewards[t:])
                n[s, a] += 1
                q[s, a] = total_rewards[s, a] / n[s, a]

    return pd.merge(
        pd.DataFrame(q.items(), columns=['s, a', 'value']),
        pd.DataFrame(n.items(), columns=['s, a', 'n']),
        on='s, a'
    )


def q_learning(n_iter=10_000, alpha=0.2, gamma=0.99):

    q = np.zeros([n_states, n_actions])

    def e_greedy(s, e=0.1):
        if np.random.uniform(0, 1) < e or np.sum(np.abs(s)) == 0:
            return np.random.randint(0, n_actions)
        else:
            return np.argmax(q[s])

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


def mab(n_iter=10_000):

    from data import bandit_bruteforce
    env = bandit_bruteforce.BanditTwoArmedHighLowFixed()
    n_actions = env.action_space.n
    # print(env.p_dist)  # [0.8, 0.2]
    q = np.zeros(n_actions)
    sum_rewards = np.zeros(n_actions)
    n = np.zeros(n_actions)

    def soft_max(i):
        denominator = np.sum([np.exp(q[a] * i/n_iter) for a in range(n_actions)])
        probas = [np.exp(q[a] * i/n_iter) / denominator for a in range(n_actions)]
        return np.random.choice(n_actions, p=probas)

    def ucb(i):
        if i < n_actions:
            return i
        else:
            return np.argmax([
                q[a] + (2*np.log(i) / n[a]) for a in range(n_actions)
            ])
    env.reset()
    for i in range(n_iter):
        a = soft_max(i)
        _, r, _, _ = env.step(a)
        sum_rewards[a] += r
        n[a] += 1
        q[a] = sum_rewards[a] / n[a]
    return q


# mab = mab()


##################################################


np.random.seed(1)


class NN:  # hardcoded for classification

    def __init__(self, eta=0.001, n_epoch=1_000, n_hidden=10, batch_size=10, classification=True):
        self.eta = eta
        self.n_epoch = n_epoch
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
    def activation_sigmoid(Z):
        return 1 / (1 + np.exp(-np.clip(Z, a_min=-250, a_max=250)))

    @staticmethod
    def derivative_sigmoid(Z):
        return Z * (1 - Z)

    def forward(self, X):
        if self.classification:
            Z_h = self.b_h + np.dot(X, self.W_h)
            A_h = self.activation_sigmoid(Z_h)
            Z_o = self.b_o + np.dot(A_h, self.W_o)
            A_o = self.activation_sigmoid(Z_o)
            return Z_h, A_h, Z_o, A_o

    def predict(self, X):
        _, _, Z_o, _ = self.forward(X)
        return Z_o

    def fit(self, X, y, l2=0):
        if self.classification:
            y = self.onehot(y)  # will become a matrix y, not vector
        self.b_h = np.zeros(self.n_hidden)
        self.W_h = np.zeros([X.shape[1], self.n_hidden])
        self.b_o = np.zeros(y.shape[1])
        self.W_o = np.zeros([self.n_hidden, y.shape[1]])
        for i in range(self.n_epoch):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            for idx in range(0, X.shape[0]-self.batch_size, self.batch_size):
                batch = indices[idx: idx+self.batch_size]
                Z_h, A_h, Z_o, A_o = self.forward(X[batch])
                delta_o = y[batch] - A_o
                delta_h = np.dot(delta_o, self.W_o.T) * self.derivative_sigmoid(A_h)
                self.b_o = self.b_o + self.eta * np.sum(delta_o, axis=0)
                self.W_o = self.W_o + self.eta * np.dot(A_h.T, delta_o) + self.W_o * l2
                self.b_h = self.b_h + self.eta * np.sum(delta_h, axis=0)
                self.W_h = self.W_h + self.eta * np.dot(X[batch].T, delta_h) + self.W_h * l2
            print('Epoch: %d' % i)
        return self


iris = pd.read_csv('data/iris.csv')
X_train, X_test, y_train, y_test = train_test_split(
    iris.iloc[:, 0:-1].values,
    LabelEncoder().fit_transform(iris.iloc[:, -1].values),
    test_size=0.5
)
nn = NN(n_epoch=10_000, eta=0.001)
nn.fit(X_train, y_train)
prediction = np.argmax(nn.predict(X_test), axis=1)
print(
    'Accuracy: %f' % (np.sum(prediction == y_test) / y_test.shape[0])
)



























































































































































































































































































































































