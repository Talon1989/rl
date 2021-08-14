import random
import gym
import numpy as np
from collections import deque
import tensorflow
from tensorflow import keras


class CustomPriorityDqnCart:

    def __init__(self, n_episodes, gamma, batch_size, weights_location):
        self.n_episodes = n_episodes
        self.env = gym.make('CartPole-v1')  # v0 = max reward 200 ; v1 = max reward 500
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.gamma = gamma
        self.batch_size = batch_size
        self.weights_location = weights_location
        self.epsilon = 1.
        self.epsilon_decay = 995/1000
        self.epsilon_min = 1/1000
        self.replay_buffer = deque(maxlen=100)
        self.beta = 1
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

    def calculate_priority_0(self, s, a, r, s_):
        '''
        :return: lower delta has lower priority (sort descending)
        '''
        delta = r + self.gamma * np.max(self.target_network.predict(s_)) - self.main_network.predict(s)[0][a]
        delta = np.abs(delta) + 1
        return delta

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def save_main_nn_weights(self):
        self.main_network.save_weights(self.weights_location)

    def load_main_nn_weights(self):
        self.main_network.load_weights(self.weights_location)

    def e_greedy(self, s):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            # return np.argmax(self.main_network.predict(s))
            return np.argmax(self.main_network.predict(s)[0])

    def store_transition(self, s, a, r, s_, done):
        priority = self.calculate_priority_0(s, a, r, s_)
        self.replay_buffer.append([s, a, r, s_, done, priority])

    def store_transition_sorted_scratch(self, s, a, r, s_, done):
        # np.random.choice(['a', 'b', 'c'], size=3, replace=False, p=[0.1, 0.5, 0.4])
        priority = self.calculate_priority_0(s, a, r, s_)
        self.replay_buffer.append([s, a, r, s_, done, priority])
        priorities = np.array(self.replay_buffer)[:, -1].astype(float)
        denom = np.sum(priorities)
        weighted_priorities = priorities / denom
        vanilla_indices = np.random.choice(
            np.arange(self.batch_size),
            size=self.batch_size,
            replace=False,
            p=weighted_priorities
        )

    def get_batch_from_sorted_buffer(self):
        priorities = np.array(self.replay_buffer)[:, -1].astype(float)
        denom = np.sum(np.exp(priorities * (1 - self.beta)))
        weighted_priorities = np.exp(priorities * (1 - self.beta)) / denom
        vanilla_indices = np.random.choice(
            np.arange(100),
            # size=self.batch_size,
            size=100,
            replace=False,
            p=weighted_priorities
        )
        self.beta = self.beta * 995/1000
        return np.array([
            x for _, x in sorted(zip(vanilla_indices, self.replay_buffer))
        ])[0: self.batch_size]


    def store_transition_sorted(self, s, a, r, s_, done):
        # np.random.choice(['a', 'b', 'c'], size=3, replace=False, p=[0.1, 0.5, 0.4])
        priority = self.calculate_priority_0(s, a, r, s_)
        self.replay_buffer.append([s, a, r, s_, done, priority])

    def replay(self):
        importance_weight = lambda x: ((1/self.batch_size) * x[-1]) ** (1 - 1/self.beta)
        self.beta = self.beta * 995/1000
        minibatch = sorted(self.replay_buffer, key=importance_weight, reverse=True)[0: self.batch_size]
        # print(np.array(minibatch)[:, -1])
        for s, a, r, s_, done, _ in minibatch:
            if done:
                target = r
            else:
                target = r + self.gamma * np.max(self.main_network.predict(s_)[0])
            q_values = self.main_network.predict(s)
            q_values[0][a] = target
            self.main_network.fit(s, q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def replay_2(self):
        # importance_weight = lambda x: ((1/self.batch_size) * x[-1]) ** (1 - 1/self.beta)
        minibatch = self.get_batch_from_sorted_buffer()
        # minibatch = self.sorted_replay_buffer[0: self.batch_size]
        # minibatch = np.array(sorted(self.replay_buffer, key=lambda x: x[-1], reverse=False)[0: self.batch_size])
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
        targets = rewards + self.gamma * np.amax(self.target_network.predict(states_), axis=1) * (1 - dones)
        q_values = self.main_network.predict(states)
        for i in range(q_values.shape[0]):
            q_values[i, actions[i]] = targets[i]
        self.main_network.fit(states, q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay

    def train(self):
        scores = []
        for episode in range(self.n_episodes):
            s = self.env.reset()
            s = np.reshape(s, [1, self.n_states])
            for i in range(100_000):
                self.env.render()
                a = self.e_greedy(s)
                s_, r, done, _ = self.env.step(a)
                r = r if not done else -100
                s_ = np.reshape(s_, [1, self.n_states])
                # self.store_transition(s, a, r, s_, done)
                self.store_transition_sorted(s, a, r, s_, done)
                # if len(self.replay_buffer) >= self.batch_size and i % 4 == 0:
                #     self.replay()
                if done:
                    scores.append(i)
                    print('Episode %d , epsilon: %.2f , beta: %.2f , score: %d , avg scores: %.2f'
                          % (episode, self.epsilon, self.beta,  i, np.sum(scores) / len(scores)))
                    break
                s = s_
            if len(self.replay_buffer) >= 100:
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



dqn = CustomPriorityDqnCart(3000, 0.95, 2**5, 'data/custom_dqn_cartpole/prioritized_dqn_weights.hdf5')
dqn.train()
print()


















































































































































































































































































































































