import gym
import numpy as np
from tensorflow import keras
import tensorflow as tf


#  https://medium.com/swlh/policy-gradient-reinforcement-learning-with-keras-57ca6ed32555


def onehot(y):
    Y = np.zeros([np.unique(y).shape[0], y.shape[0]])
    for idx, val in enumerate(y):
        Y[int(val), idx] = 1
    return Y.T


np.random.seed(1)
tf.random.set_seed(1)
n_episodes = 1_000
env = gym.make('CartPole-v1')  # env to import
env.seed(1)
env.reset()


class Reinforce:

    def __init__(self, env: gym.wrappers.time_limit.TimeLimit, weight_path: str):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.gamma = 0.99
        self.alpha = 1e-4
        self.eta = 0.01
        self.weight_path = weight_path
        self.states, self.gradients, self.rewards = [], [], []
        self.probas, self.discounted_rewards, self.total_rewards = [], [], []
        self.model = self._create_model()

    def hot_encode_action(self, action):
        action_encoded = np.zeros(self.action_size, np.float32)
        action_encoded[action] = 1
        return action_encoded

    def remember(self, s, a, a_proba, r):
        self.gradients.append(self.hot_encode_action(a) - a_proba)
        self.states.append(s)
        self.rewards.append(r)
        self.probas.append(a_proba)

    def _create_model(self):
        model = keras.models.Sequential([
            keras.layers.Dense(units=2**5, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(units=2**4, activation='relu'),
            keras.layers.Dense(units=self.action_size, activation='softmax')
        ])
        model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=self.eta)
        )
        return model

    #  CHECK BETTER
    def get_action(self, s):
        s = s.reshape([1, s.shape[0]])
        action_proba_distr = self.model.predict(s).flatten()
        action_proba_distr = action_proba_distr / np.sum(action_proba_distr)
        a = np.random.choice(self.action_size, 1, p=action_proba_distr)[0]
        return a, action_proba_distr

    def get_discounted_rewards(self, rewards):
        discounted_rewards = []
        cumulative_total_return = 0
        for r in rewards[::-1]:
            cumulative_total_return = r + (self.gamma * cumulative_total_return)
            discounted_rewards.insert(0, cumulative_total_return)
            # normalization
            mean_rewards = np.mean(discounted_rewards)
            std_rewards = np.std(discounted_rewards) + 1e-7  # to avoid zero division
            return (discounted_rewards - mean_rewards) / std_rewards

    def update_policy(self):
        states = np.vstack(self.states)
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        discounted_rewards = self.get_discounted_rewards(rewards)
        gradients = gradients * discounted_rewards
        gradients = self.alpha * np.vstack([gradients]) + self.probas
        self.states, self.probas, self.gradients, self.rewards = [], [], [], []
        return self.model.fit(states, gradients, epochs=1)
        # history = self.model.train_on_batch(states, gradients)

    def train(self, n_episodes, rollout_period=1, render_period=50):
        env = self.env
        total_rewards = np.zeros(n_episodes)
        for ep in range(n_episodes):
            s = env.reset()
            ep_reward = 0
            while True:
                if ep % render_period == 0:
                    env.render()
                a, proba = self.get_action(s)
                s_, r, done, _ = env.step(a)
                self.remember(s, a, proba, r)
                ep_reward += r

                if done:
                    if ep % rollout_period == 0:
                        print('Updating policy at episode %d' % ep)
                        self.update_policy()
                    break
                s = s_
            print('Episode %d ; Reward %d' % ((ep + 1), ep_reward))
            total_rewards[ep] = ep_reward
        self.total_rewards = total_rewards






























































































































































































































































































































































































































































