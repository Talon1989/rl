# import gym
# import numpy as np
# from tensorflow import keras
# # from keras import backend as K
# # from keras import utils as np_utils
#
#
# #  https://gist.github.com/kkweon/c8d1caabaf7b43317bc8825c226045d2
#
#
# env = gym.make('CartPole-v1')
#
#
# class Agent:
#
#     def __init__(self, state_size, n_actions, hidden_dims=None):
#         if hidden_dims is None:
#             hidden_dims = [2**5, 2**5]
#         self.state_size = state_size
#         self.n_actions = n_actions
#         self.hidden_dims = hidden_dims
#         self.X = None
#         self.model = self.build_network()
#         self.train_fn = None
#         # self.model = None
#         # self.build_network()
#         self.__build_train_fn_og()
#
#     def __build_network_og(self):
#         self.X = keras.layers.Input(shape=(self.state_size, ))
#         net = self.X
#         for h_dim in self.hidden_dims:
#             net = keras.layers.Activation('relu')(keras.layers.Dense(h_dim)(net))
#         net = keras.layers.Activation('softmax')(keras.layers.Dense(self.n_actions)(net))
#         self.model = keras.models.Model(inputs=self.X, outputs=net)
#
#     #  hardcoded for 2 hidden layers of 32 units each
#     def build_network(self):
#         return keras.models.Sequential([
#             keras.layers.Dense(units=32, input_dim=self.state_size, activation='relu'),
#             keras.layers.Dense(units=32, activation='relu'),
#             keras.layers.Dense(units=self.n_actions, activation='softmax')
#         ])
#
#     def __build_train_fn_og(self):
#         """Create a train function
#         It replaces `model.fit(X, y)` because we use the output of model and use it for training.
#         For example, we need action placeholder
#         called `action_one_hot` that stores, which action we took at state `s`.
#         Hence, we can update the same action.
#         This function will create
#         `self.train_fn([state, action_one_hot, discount_reward])`
#         which would train the model.
#         """
#         action_proba_placeholder = self.model.output
#         action_onehot_placeholder = keras.backend.placeholder(
#             shape=(None, self.n_actions), name='action_onehot'
#         )
#         discount_reward_placeholder = keras.backend.placeholder(
#             shape=(None, ), name='discount_reward'
#         )
#         action_proba = keras.backend.sum(
#             action_proba_placeholder * discount_reward_placeholder, axis=1
#         )
#         loss = keras.backend.log(action_proba) * discount_reward_placeholder
#         loss = keras.backend.mean(loss)
#         updates = keras.optimizers.Adam().get_updates(
#             params=self.model.trainable_weights, loss=loss
#         )
#         self.train_fn = keras.backend.function(
#             inputs=[self.model.input, action_onehot_placeholder, discount_reward_placeholder],
#             outputs=[],
#             updates=updates
#         )
#
#     def get_action(self, s):
#         shape = s.shape
#         if len(shape) == 1:
#             assert shape == (self.state_size, ), "{} != {}".format(shape, self.state_size)
#             s = np.expand_dims(s, axis=0)
#         elif len(shape) == 2:
#             assert shape[1] == self.state_size, "{} != {}".format(shape, self.state_size)
#         else:
#             raise TypeError("Wrong state shape is given: {}".format(s.shape))
#         action_proba = np.squeeze(self.model.predict(s))
#         assert len(action_proba) == self.n_actions, "{} != {}".format(len(action_proba), self.n_actions)
#         return np.random.choice(np.arange(self.n_actions), p=action_proba)
#
#     def fit(self, s, a, r):
#         action_onehot = keras.utils.to_categorical(a, num_classes=self.n_actions)
#         discount_reward = self.compute_discounted_reward(r)
#         assert s.shape[1] == self.state_size, "{} != {}".format(s.shape[1], self.state_size)
#         assert action_onehot.shape[0] == s.shape[0], "{} != {}".format(action_onehot.shape[0], s.shape[0])
#         assert action_onehot.shape[1] == self.n_actions, "{} != {}".format(action_onehot.shape[1], self.n_actions)
#         assert len(discount_reward.shape) == 1, "{} != 1".format(len(discount_reward.shape))
#         self.train_fn([s, action_onehot, discount_reward])
#
#     @staticmethod
#     def compute_discounted_reward(r, discount_rate=0.99):
#         discounted_r = np.zeros_like(r, dtype=np.float32)
#         running_add = 0
#         for t in reversed(range(len(r))):
#             running_add = running_add * discount_rate + r[t]
#             discounted_r[t] = running_add
#         std = np.std(discounted_r)
#         discounted_r = discounted_r - discounted_r.mean()
#         discounted_r = discounted_r / std
#         return discounted_r
#
#
# def run_episode(env: gym.wrappers.time_limit.TimeLimit, agent: Agent):
#     states, actions, rewards = [], [], []
#     s = env.reset()
#     total_reward = 0
#     while True:
#         a = agent.get_action(s)
#         s_, r, done, _ = env.step(a)
#         total_reward += r
#         states.append(s)
#         actions.append(a)
#         rewards.append(r)
#         if done:
#             agent.fit(
#                 np.array(states),
#                 np.array(actions),
#                 np.array(rewards)
#             )
#             break
#         s = s_
#     return total_reward
#
#
# n_episodes = 2_000
#
#
# def main():
#     env = gym.make('CartPole-v1')
#     input_dimension = env.observation_space.shape[0]
#     output_dimension = env.action_space.n
#     agent = Agent(state_size=input_dimension, n_actions=output_dimension, hidden_dims=[16, 16])
#     for ep in range(n_episodes):
#         reward = run_episode(env, agent)
#         print('Episode: %d ; Reward %d' % (ep, reward))
#
#
# # agent = Agent(env.observation_space.shape[0], env.action_space.n)
# main()


#  https://keras.io/examples/rl/ddpg_pendulum/
#  Deep Deterministic Policy Gradient
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# In 'Pendulum-v0' actions are continuous instead of being discrete.
# That is, instead of using two discrete actions like -1 or +1,
# we have to select from infinite actions ranging from -2 to +2.
env = gym.make('Pendulum-v0')

def random_test():
    for i in range(10):
        s = env.reset()
        episode_reward = 0
        while True:
            env.render()
            _, r, done, _ = env.step(np.random.uniform(-2, 2, 2))
            episode_reward += r
            if done:
                print('Episode %d ; Reward %.2f' % (i+1, episode_reward))
                break

n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
action_value_upper_bound = env.action_space.high[0]
action_value_lower_bound = env.action_space.low[0]


#  Ornstein-Uhlenbeck process: https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
class OUActionNoise:

    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.x_prev = None
        self.reset()

    def __call__(self):
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)














































































































































































































































