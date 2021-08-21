import random
import gym
import numpy as np
from collections import deque
from tensorflow import keras
import matplotlib.pyplot as plt
import os


def value_iteration():
    env = gym.make('FrozenLake-v0')
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    real_target = np.array([0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0])
    gamma = 0.99
    P = env.P
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
        if np.sum(np.abs(old_v - v)) < 1e-16:
            break
    policy = np.zeros(n_states)
    for s in range(n_states):
        policy[s] = np.argmax([
            np.sum([
                (r + gamma * v[s_]) * p
                for p, s_, r, _ in P[s][a]
            ]) for a in range(n_actions)
        ])
    return policy


# pol = value_iteration()


def q_learning(n_episodes=1_000):
    env = gym.make('FrozenLake-v0')
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    real_target = np.array([0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0])
    alpha = 0.1
    gamma = 0.99
    epsilon = 1
    q = np.zeros([n_states, n_actions])
    def e_greedy(s):
        if np.random.uniform(0, 1) < epsilon or np.sum(s) == 0:
            return np.random.randint(0, n_actions)
        else:
            return np.argmax(s)
    for e in range(n_episodes):
        s = env.reset()
        while True:
            a = e_greedy(s)
            if epsilon > 0.1:
                epsilon *= 999/1000
            s_, r, done, _ = env.step(a)
            q[s, a] += alpha * ((r + gamma * np.max(q[s_])) - q[s, a])
            if done:
                break
            s = s_
    return q


# q = q_learning(10_000)


env = gym.make('CartPole-v1')


class DQL:

    def __init__(
            self, n_episodes=2_000, alpha=0.001, gamma=0.95,
            epsilon_decay=0.995, batch_size=2**5
    ):
        self.env = gym.make('CartPole-v1')
        self.n_actions = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]
        self.n_episodes = n_episodes
        self.learning_rate = alpha
        self.gamma = gamma
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 1/100
        self.batch_size = batch_size
        self.main_nn = self.build_nn()
        self.target_nn = self.build_nn()
        self.replay_buffer = deque(maxlen=1_000)

    def build_nn(self):
        model = keras.models.Sequential([
            keras.layers.Dense(units=2**6, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(units=2**6, activation='relu'),
            keras.layers.Dense(units=self.n_actions, activation='linear')
        ])
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )
        return model

    def update_target_nn(self):
        self.target_nn.set_weights(self.main_nn.get_weights())

    def save_weights(self, path):
        self.main_nn.save_weights(path)

    def load_weights(self, path):
        self.main_nn.load_weights(path)

    def remember(self, s, a, r, s_, done):
        self.replay_buffer.append([s, a, r, s_, done])

    def get_updated_minibatch(self):
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
        return states, actions, rewards, states_, dones

    def e_greedy(self, s):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            return np.argmax(self.main_nn.predict(s))

    def replay(self):
        states, actions, rewards, states_, dones = self.get_updated_minibatch()
        targets = rewards + self.gamma * np.max(self.target_nn.predict(states_), axis=1) * (1 - dones)
        q_values = self.main_nn.predict(states)
        for i in range(self.batch_size):
            q_values[i, actions[i]] = targets[i]
        self.main_nn.fit(states, q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def train(self):
        scores = []
        avg_scores = []
        for e in range(self.n_episodes):
            s = self.env.reset()
            s = s.reshape([1, self.state_size])
            score = 0
            while True:
                # self.env.render()
                a = self.e_greedy(s)
                s_, r, done, _ = self.env.step(a)
                r = r * (1 - done) - 100 * done
                s_ = s_.reshape([1, self.state_size])
                self.remember(s, a, r, s_, done)
                if done:
                    scores.append(score)
                    avg_scores.append(np.sum(scores) / len(scores))
                    print('Episode: %d , epsilon: %.3f , score: %d , avg score: %.3f'
                          % (e, self.epsilon, score, avg_scores[-1]))
                    break
                s = s_
                score += 1
            if len(self.replay_buffer) >= self.batch_size:
                self.replay()
            if e > 0 and e % 10 == 0:
                print('\nUpdating target network\n')
                self.update_target_nn()
            if e > 0 and e % 100 == 0:
                print('\nStoring main nn weights in local memory\n')
                plt.plot(scores, c='r', linewidth=1, label='scores')
                plt.plot(avg_scores, c='b', linewidth=1, label='avg scores')
                plt.title('EP: %d , avg scores over episodes' % e)
                plt.xlabel('episodes')
                plt.ylabel('score')
                plt.legend(loc='best')
                plt.show()
                plt.clf()
                self.save_weights('data/zzz_train_q_ql_dql_/main_weight.hdf5')

        return self

    def test(self, n_ep):
        self.epsilon = 0
        self.load_weights('data/zzz_train_q_ql_dql_/main_weight.hdf5')
        for e in range(self.n_episodes):
            s = self.env.reset()
            s = s.reshape([1, self.state_size])
            score = 0
            while True:
                self.env.render()
                a = self.e_greedy(s)
                s_, r, done, _ = self.env.step(a)
                s_ = s_.reshape([1, self.state_size])
                if done:
                    print('Episode: %d , score: %d'
                          % (e, score))
                    break
                s = s_
                score += 1


if not os.path.exists('data/zzz_train_q_ql_dql_/'):
    os.makedirs('data/zzz_train_q_ql_dql_/')
dql = DQL(n_episodes=3_000)
# dql.load_weights('data/zzz_train_q_ql_dql_/main_weight.hdf5')
# dql.epsilon = 2/10
# dql.target_nn.load_weights('data/zzz_train_q_ql_dql_/main_weight.hdf5')
# dql.train()
dql.test(10)



















































































































































































































































































































































































































































































