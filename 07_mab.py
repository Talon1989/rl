import numpy as np
import pandas as pd
import gym
from data import bandit_bruteforce


env = bandit_bruteforce.BanditTwoArmedHighLowFixed()
n_actions = env.action_space.n
# print(env.p_dist)  # [0.8, 0.2]


#  E-GREEDY  ############################


count = np.zeros(n_actions)
sum_rewards = np.zeros(n_actions)
q = np.zeros(n_actions)
n_rounds = 100_000


def epsilon_greedy(epsilon=0.1):
    if np.random.uniform(0, 1) < epsilon or np.sum(np.abs(q)) == 0:
        return np.random.randint(0, n_actions)
    else:
        return np.argmax(q)


# for i in range(n_rounds):
#     action = epsilon_greedy(0.5)
#     _, r, _, _ = env.step(action)
#     count[action] += 1
#     sum_rewards[action] += r
#     q[action] = sum_rewards[action] / count[action]


#  SOFTMAX  ############################


count = np.zeros(n_actions)
sum_rewards = np.zeros(n_actions)
q = np.zeros(n_actions)
n_rounds = 1_000


def softmax(T):
    denom = np.sum([
        np.exp(i/T) for i in q
    ])
    p = [np.exp(i/T) / denom for i in q]
    arm = np.random.choice(n_actions, p=p)
    return arm


T = 50


# for i in range(n_rounds):
#     a = softmax(T)
#     s_, r, done, _ = env.step(a)
#     sum_rewards[a] += r
#     count[a] += 1
#     q[a] = sum_rewards[a] / count[a]
#     if T > 1:
#         T = T * 99/100


#  CUSTOM SOFTMAX  ############################


count = np.zeros(n_actions)
sum_rewards = np.zeros(n_actions)
q = np.zeros(n_actions)
n_rounds = 1_000


def softmax(T):
    denom = np.sum([
        np.exp(i*T) for i in q
    ])
    p = [np.exp(i*T) / denom for i in q]
    arm = np.random.choice(n_actions, p=p)
    return arm


# for i in range(n_rounds):
#     a = softmax(T=i/n_rounds)
#     _, r, _, _ = env.step(a)
#     sum_rewards[a] += r
#     count[a] += 1
#     q[a] = sum_rewards[a] / count[a]


#  UCB  ############################


count = np.zeros(n_actions)
sum_rewards = np.zeros(n_actions)
q = np.zeros(n_actions)
n_rounds = 1_000


def UCB(i):
    """"
    :param i: # iteration
    :return:
    """
    ucb = np.zeros(2)
    if i < n_actions:  # we force to explore each arm at least ones at the beginning
        return i
    for arm in range(n_actions):
        ucb[arm] = q[arm] + np.sqrt(
            (2*np.log(np.sum(count))) / count[arm]
        )
    return np.argmax(ucb)


for i in range(n_rounds):
    a = UCB(i)
    _, r, _, _ = env.step(a)
    count[a] += 1
    sum_rewards[a] += r
    q[a] = sum_rewards[a] / count[a]


#  Thompson sampling






























































































































































































































































































































































































































































