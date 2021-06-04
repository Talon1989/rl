import numpy as np
import pandas as pd
import gym
from data import bandit_bruteforce


env = bandit_bruteforce.BanditTwoArmedHighLowFixed()
n_actions = env.action_space.n
# print(env.p_dist)  # [0.8, 0.2]


count = np.zeros(n_actions)
sum_rewards = np.zeros(n_actions)
q = np.zeros(n_actions)
n_rounds = 100_000


def epsilon_greedy(epsilon=0.1):
    if np.random.uniform(0, 1) < epsilon or np.sum(np.abs(q)) == 0:
        return np.random.randint(0, n_actions)
    else:
        return np.argmax(q)


for i in range(n_rounds):
    action = epsilon_greedy(0.5)
    _, r, _, _ = env.step(action)
    count[action] += 1
    sum_rewards[action] += r
q = sum_rewards / count















