import random
import gym
import numpy as np
from collections import deque
import tensorflow
from tensorflow import keras


env = gym.make('CartPole-v1')  # v0 = max reward 200 ; v1 = max reward 500
s_size = env.observation_space.shape[0]
a_size = env.action_space.n


class CustomDqn:

    def __init__(self, n_iter, state_size, action_size, alpha, gamma, batch_size):
        self.n_iter = n_iter
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.
        self.epsilon_decay = 999/1000
        self.epsilon_min = 1/1000
        self.replay_buffer = deque(maxlen=1_000)
        self.main_network = self.build_nn()
        self.target_network = self.build_nn()

    def build_nn(self):
        model = keras.models.Sequential([
            keras.layers.Dense(2**5, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(2**5, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(lr=1/1000),
            metrics=['accuracy']
        )
        return model

    @staticmethod
    def copy_nn(nn_from, nn_to):
        nn_to.set_weights(nn_from.get_weights())


dqn = CustomDqn(10_000, s_size, a_size, 0.01, 0.99, 2**6)

















































































































































































































































































































































































































