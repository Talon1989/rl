import random
import gym
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import os


###################### DQL without deque collection ######################


env = gym.make('CartPole-v1')


class ReplayBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.states, self.actions, self.rewards, self.next_states, self.dones = [], [], [], [], []

    def remember(self, s, a, r, s_, done):
        if len(self.states) == self.max_size:
            del self.states[0]
            del self.actions[0]
            del self.rewards[0]
            del self.next_states[0]
            del self.dones[0]
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.next_states.append(s_)
        self.dones.append(done)

    def get_buffer(self, batch_size):
        assert batch_size < self.max_size
        indices = np.arange(self.get_buffer_size())
        np.random.shuffle(indices)
        pre_buffer_states = [self.states[i] for i in indices][0: batch_size]
        buffer_actions = [self.actions[i] for i in indices][0: batch_size]
        buffer_rewards = [self.rewards[i] for i in indices][0: batch_size]
        pre_buffer_next_states = [self.next_states[i] for i in indices][0: batch_size]
        buffer_dones = [self.dones[i] for i in indices][0: batch_size]
        buffer_states, buffer_next_states = [], []
        for i in range(len(pre_buffer_states)):
            buffer_states.append(pre_buffer_states[i][0])
            buffer_next_states.append(pre_buffer_next_states[i][0])
        return np.array(buffer_states), np.array(buffer_actions), np.array(buffer_rewards),\
               np.array(buffer_next_states), np.array(buffer_dones)

    def get_buffer_size(self):
        return len(self.actions)


buffer = ReplayBuffer(max_size=50)
for i in range(200):
    buffer.remember(
        np.random.normal(0, 1, 4),
        np.random.randint(0, 4),
        np.random.uniform(-2, 10),
        np.random.normal(0, 1, 4),
        np.random.binomial(1, 0.5)
    )
states, actions, rewards, states_, dones = buffer.get_buffer(10)


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
        self.replay_buffer = ReplayBuffer(max_size=1_000)

    def build_nn(self):
        model = keras.models.Sequential([
            keras.layers.Dense(units=2**5, input_shape=(self.state_size, ), activation='relu'),
            keras.layers.Dense(units=2**5, activation='relu'),
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
        self.replay_buffer.remember(s, a, r, s_, done)

    def e_greedy(self, s):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            return np.argmax(self.main_nn.predict(s))

    def replay(self):
        states, actions, rewards, states_, dones = self.replay_buffer.get_buffer(self.batch_size)
        # print()
        # print(states.shape)
        # print(actions.shape)
        # print(rewards.shape)
        # print(states_.shape)
        # print(dones.shape)
        # print()
        if self.epsilon > 0.05:
            targets = rewards + self.gamma * (1 - self.epsilon) * np.max(self.target_nn.predict(states_), axis=1) * (1 - dones)
        else:
            targets = rewards + self.gamma * np.max(self.target_nn.predict(states_), axis=1) * (1 - dones)
        q_values = self.main_nn.predict(states)
        for i in range(self.batch_size):
            q_values[i][actions[i]] = targets[i]
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
            if (self.replay_buffer.get_buffer_size()) >= self.batch_size:
                # print(self.replay_buffer.get_buffer_size())
                # print(self.batch_size)
                self.replay()
            if e > 0 and e % 10 == 0:
                print('\nUpdating target network\n')
                self.update_target_nn()
            if (e+1) % 100 == 0:
                print('\nStoring main nn weights in local memory\n')
                plt.plot(scores, c='r', linewidth=1, label='scores')
                plt.plot(avg_scores, c='b', linewidth=1, label='avg scores')
                plt.title('EP: %d , avg scores over episodes' % e)
                plt.xlabel('episodes')
                plt.ylabel('score')
                plt.legend(loc='best')
                plt.show()
                plt.clf()
                self.save_weights('data/zzz_train_3_custom_dql/main_weight.hdf5')
        return self


if not os.path.exists('data/zzz_train_3_custom_dql/'):
    os.makedirs('data/zzz_train_3_custom_dql/')

dql = DQL(n_episodes=1_000)
dql.train()






























































































































































































































































































































































































































































































