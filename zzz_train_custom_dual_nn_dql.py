import random
import gym
import numpy as np
from collections import deque
import tensorflow
from tensorflow import keras


class CustomDqnCart:

    def __init__(self, n_episodes, gamma, batch_size):
        self.n_episodes = n_episodes
        self.env = gym.make('CartPole-v1')  # v0 = max reward 200 ; v1 = max reward 500
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.
        self.epsilon_decay = 995/1000
        self.epsilon_min = 1/1000
        self.replay_buffer = deque(maxlen=1_000)
        self.main_network = self.build_nn()
        self.target_network = self.build_nn()

    def build_nn(self):
        model = keras.models.Sequential([
            keras.layers.Dense(2**5, input_dim=self.n_states, activation='relu'),
            keras.layers.Dense(2**5, activation='relu'),
            keras.layers.Dense(self.n_actions, activation='linear')
        ])
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(lr=1/1000),
            metrics=['accuracy']
        )
        return model

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def save_main_nn_weights(self):
        self.main_network.save_weights('data/custom_dqn_cartpole/main_weight.hdf5')

    def load_main_nn_weights(self):
        self.main_network.load_weights('data/custom_dqn_cartpole/main_weight.hdf5')

    def e_greedy(self, s):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            # return np.argmax(self.main_network.predict(s))
            return np.argmax(self.main_network.predict(s)[0])

    def store_transition(self, s, a, r, s_, done):
        self.replay_buffer.append([s, a, r, s_, done])

    def replay(self):
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        for s, a, r, s_, done in minibatch:
            if done:
                target = r
            else:
                # target = r + self.gamma * np.max(self.target_network.predict(s_))
                # target = r + self.gamma * np.max(self.target_network.predict(s_)[0])
                target = r + self.gamma * np.max(self.main_network.predict(s_)[0])
            q_values = self.main_network.predict(s)
            # q_values[a] = target
            q_values[0][a] = target
            self.main_network.fit(s, q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def replay_2(self):
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
        # states = np.reshape(states, [states.shape[0], self.n_states])
        # states_ = np.reshape(states_, [states_.shape[0], self.n_states])
        targets = rewards + self.gamma * np.amax(self.target_network.predict(states_), axis=1) * (1 - dones)
        q_values = self.main_network.predict(states)
        # q_values[:, actions] = targets
        for i in range(q_values.shape[0]):
            q_values[i, actions[i]] = targets[i]
        self.main_network.fit(states, q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def train(self):
        for episode in range(self.n_episodes):
            s = self.env.reset()
            s = np.reshape(s, [1, self.n_states])
            for i in range(100_000):
                self.env.render()
                a = self.e_greedy(s)
                s_, r, done, _ = self.env.step(a)
                r = r if not done else -100
                s_ = np.reshape(s_, [1, self.n_states])
                self.store_transition(s, a, r, s_, done)
                # if len(self.replay_buffer) >= self.batch_size and i % 4 == 0:
                #     self.replay()
                if done:
                    print('Episode %d , epsilon: %.2f , score: %d' % (episode, self.epsilon, i))
                    break
                s = s_
            if len(self.replay_buffer) >= self.batch_size:
                # self.replay()
                self.replay_2()
            if episode > 0 and episode % 10 == 0:
                print('\nUpdating target network\n')
                self.update_target_network()
            if episode > 0 and episode % 100 == 0:
                print('\nSaving main network weights\n')
                self.save_main_nn_weights()


    def test(self):
        self.load_main_nn_weights()
        self.epsilon = 0.
        for episode in range(self.n_episodes):
            s = self.env.reset()
            s = np.reshape(s, [1, self.n_states])
            for i in range(100_000):
                self.env.render()
                a = self.e_greedy(s)
                s_, _, done, _ = self.env.step(a)
                if done:
                    print('Episode %d , score: %d' % (episode, i))
                    break
                s = np.reshape(s_, [1, self.n_states])





dqn = CustomDqnCart(3000, 0.95, 2**5)
dqn.test()

















































































































































































































































































































































































































