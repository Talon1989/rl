import random
import gym
import numpy as np
from collections import deque
from tensorflow import keras
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

    def __init__(self):
        #  TODO
        pass








































































































































































































































































































































































































































































