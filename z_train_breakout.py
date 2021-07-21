import gym
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow import keras
from baselines_bruteforce.atari_wrappers import make_atari, wrap_deepmind


# env = gym.make('BreakoutNoFrameskip-v4')
# n_actions = env.action_space.n
# color = np.array([210, 164, 74]).mean()
#
#
# def preprocess_state(s):
#     image = s[1:176:2, ::2]
#     image = image.mean(axis=2)
#     image[image == color] = 0
#     image = ((image - 128) / 128) - 1
#     image = np.expand_dims(image.reshape(88, 80, 1), axis=0)
#     return image
#
#
# def random_play(n_games=1):
#     env.reset()
#     counter = 0
#     while True:
#         action = np.random.randint(0, n_actions)
#         _, _, done, _ = env.step(action)
#         counter += done
#         env.render()
#         if counter == n_games:
#             break
#         if done:
#             print('Ending game %d' % counter)
#             env.reset()


# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (  # Rate at which to reduce chance of random action being taken
        epsilon_max - epsilon_min
)
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000


# Use the Baseline Atari environment because of Deepmind helper functions
env = make_atari("BreakoutNoFrameskip-v4")
# Warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=True, scale=True)
n_actions = env.action_space.n


def q_model():
    return keras.models.Sequential([
        keras.layers.Conv2D(
            filters=32, kernel_size=8, strides=4, activation='relu',
            padding='valid', input_shape=(84, 84, 4)
        ),
        keras.layers.Conv2D(
            filters=64, kernel_size=4, strides=2, activation='relu'
        ),
        keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=1, activation='relu'
        ),
        keras.layers.Flatten(),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=n_actions, activation='linear')
    ])


model_main = q_model()
model_target = q_model()



























































































































































































































































































































































