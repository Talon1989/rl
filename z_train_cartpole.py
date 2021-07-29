import numpy as np
import gym
from tensorflow import keras
import tensorflow
from collections import deque
import random
import matplotlib.pyplot as plt


# env = gym.make('CartPole-v1')  # r=1 for each step where it doesn't fall
# n_states = env.observation_space.shape[0]
# n_actions = env.action_space.n
#
#
# def random_actions(episodes=10):
#     for episode in range(episodes):
#         score = 0
#         s = env.reset()
#         while True:
#             env.render()
#             a = np.random.randint(0, n_actions)
#             s, r, done, _ = env.step(a)
#             score += r
#             if done:
#                 break
#         print('Episode %d, score %d' % (episode+1, score))


# def build_model():
#     return keras.models.Sequential([
#         keras.layers.Flatten(input_shape=(1, n_states)),
#         keras.layers.Dense(units=24, activation='relu'),
#         keras.layers.Dense(units=24, activation='relu'),
#         keras.layers.Dense(units=n_actions, activation='linear')
#     ])


def build_model(input_shape, n_actions):
    model = keras.models.Sequential([
        keras.layers.Dense(
            units=512, activation='relu',
            input_shape=input_shape
        ),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units=n_actions, activation='linear')
    ])
    model.compile(
        loss='mse',
        optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
        # optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    print(model.summary())
    return model


def build_model_og(input_shape, action_space):
    from keras.models import Model, load_model
    from keras.layers import Input, Dense
    from keras.optimizers import Adam, RMSprop
    X_input = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs = X_input, outputs = X)
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model


class DQNAgent:

    def __init__(self, env: gym.Env, n_states, n_actions):
        #  env._max_episode_steps is hardcoded to work only with "CartPole-v1"
        self.env = env
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_episodes = 100_000
        self.memory = deque(maxlen=2_000)
        self.gamma = 0.95
        self.epsilon = 1.
        self.epsilon_min = 0.001
        self.epsilon_decay = 9_999 / 10_000
        self.batch_size = 64
        self.train_start = 1_000
        # self.model = build_model(input_shape=(self.n_states,), n_actions=self.n_actions)
        self.model = build_model_og(
            input_shape=(self.n_states,), action_space=self.n_actions
        )

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def e_greedy(self, s):
        if np.random.uniform(0, 1) <= self.epsilon:
            # return np.random.randint(0, self.n_actions)
            return random.randrange(self.n_actions)
        else:
            return np.argmax(self.model.predict(s))

    def train(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        s = np.zeros([self.batch_size, self.n_states])
        s_ = np.zeros([self.batch_size, self.n_states])
        a, r, done = [], [], []
        for i in range(self.batch_size):
            s[i] = minibatch[i][0]
            a.append(minibatch[i][1])
            r.append(minibatch[i][2])
            s_[i] = minibatch[i][3]
            done.append(minibatch[i][4])
        target = self.model.predict(s)
        target_ = self.model.predict(s_)
        for i in range(self.batch_size):
            if done[i]:
                target[i][a[i]] = r[i]
            else:
                target[i][a[i]] = r[i] + self.gamma * (np.amax(target_[i]))
        self.model.fit(s, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.model = keras.models.load_model(name)

    def save(self, name):
        self.model.save(name)

    def run(self):
        reward_memory = []
        counter = 0
        for e in range(self.n_episodes):
            s = self.env.reset()
            s = np.reshape(s, [1, self.n_states])
            # s.reshape(1, self.n_states)
            i = 0
            while True:
                self.env.render()
                a = self.e_greedy(s)
                s_, r, done, _ = self.env.step(a)
                s_ = np.reshape(s_, [1, self.n_states])
                if not done or i == self.env._max_episode_steps-1:
                    r = r
                else:
                    r = -100
                self.remember(s, a, r, s_, done)
                s = s_
                i += 1
                if i == 500:
                    counter += 1
                    print('%d times solved' % counter)
                else:
                    counter = 0
                if done:
                    reward_memory.append(i)
                    print(
                        'episode: %d/%d, score: %d, epsilon:%f'
                        % (e, self.n_episodes, i, self.epsilon)
                    )
                    if counter == 5:
                        print('Problem solved 5 times in a row\nSaving trained model')
                        self.save('data/cartpole_custom-dqn.h5')
                        return
                    break

                self.train()
        plt.plot(reward_memory, c='b')
        plt.title('Reward history')
        plt.xlabel('iterations')
        plt.ylabel('reward')
        plt.show()
        plt.clf()

    def test(self):
        self.load('data/cartpole_custom-dqn.h5')
        for e in range(self.n_episodes):
            i = 0
            s = self.env.reset()
            while True:
                s = np.reshape(s, [1, self.n_states])
                self.env.render()
                a = np.argmax(self.model.predict(s))
                s_, r, done, _ = self.env.step(a)
                s = s_
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.n_episodes, i))
                    break


environment = gym.make('CartPole-v1')
n_states = environment.observation_space.shape[0]
n_actions = environment.action_space.n
agent = DQNAgent(environment, n_states, n_actions)
agent.run()
# agent.test()
