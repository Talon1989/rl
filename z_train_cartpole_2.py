import random

import gym
import numpy as np
from collections import deque
from tensorflow import keras
import os


env = gym.make('CartPole-v1')  # v0 = max reward 200 ; v1 = max reward 500
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 2**5
n_episodes = 1000
output_directory = 'data/cartpole_2/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.
        self.epsilon_decay = 995/1_000
        self.epsilon_min = 1/100
        self.learning_rate = 1/1_000
        self.model = self.build_model()
        self.max_memory_size = 2**6

    def build_model(self):
        model = keras.models.Sequential([
            keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(lr=self.learning_rate)
        )
        return model

    def remember(self, s, a, r, s_, done):
        self.memory.append([s, a, r, s_, done])

    def e_greedy(self, s):
        if np.random.uniform(0, 1) <= self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            return np.argmax(self.model.predict(s)[0])
            # action_values = self.model.predict(s)
            # try:
            #     _ = action_values.shape[1]
            #     return np.argmax(action_values, axis=1)
            # except IndexError:
            #     return np.argmax(action_values)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for s, a, r, s_, done in minibatch:
            target = r
            if not done:
                # target = r + self.gamma * np.amax(self.model.predict(s_)[0])
                target = r + self.gamma * np.max(self.model.predict(s_)[0])
            target_ = self.model.predict(s)
            target_[0][a] = target
            self.model.fit(s, target_, epochs=1, verbose=False)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def play():

    agent = DQNAgent(state_size, action_size)

    for e in range(n_episodes):

        s = env.reset()
        s = np.reshape(s, [1, state_size])
        score = 0

        while True:

            env.render()
            a = agent.e_greedy(s)
            s_, r, done, _ = env.step(a)
            # r = r if not done else -10
            r = (r * int(1 - done)) + (int(done) * -10)
            s_ = np.reshape(s_, [1, state_size])
            agent.remember(s, a, r, s_, done)
            s = s_
            score += 1
            if done:
                print('Episode %d/%d, score: %.3f, epsilon: %.3f'
                      % (e, n_episodes, score, agent.epsilon))
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if e % 50 == 0:
            print('\nsaving weights at episode %d\n' % e)
            agent.save(output_directory + '_weights_%d_.hdf5' % e)


def test():

    agent = DQNAgent(state_size, action_size)
    agent.epsilon = agent.epsilon_min
    agent.load('data/cartpole_2/_weights_950_.hdf5')

    for e in range(n_episodes):

        s = env.reset()
        s = np.reshape(s, [1, state_size])
        score = 0

        while True:

            env.render()
            a = agent.e_greedy(s)
            s_, r, done, _ = env.step(a)
            # r = r if not done else -10
            s_ = np.reshape(s_, [1, state_size])
            s = s_
            score += 1
            if done:
                print('Episode %d, score: %d' % (e, score))
                break


play()
# test()
















































































































































































































































































































































































