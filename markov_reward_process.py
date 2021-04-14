import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import markov_decision_process


m = 3
m2 = m ** 2
P = np.zeros([m2 + 1, m2 + 1])
#  cols = to, rows = from
P[:m2, :m2] = markov_decision_process.get_P(3, 0.2, 0.3, 0.25, 0.25)
for i in range(m2):
    P[i, m2] = P[i, i]
    P[i, i] = 0
P[m2, m2] = 1


def average_rewards():
    n = 10 ** 3
    rewards = np.zeros(m2)
    for s in range(m2):
        for i in range(n):
            crashed = False
            s_next = s
            episode_reward = 0
            while not crashed:
                s_next = np.random.choice(m2 + 1, p=P[s_next, :])
                if s_next < m2:
                    episode_reward += 1
                else:  # absorbing state
                    crashed = True
            rewards[s] += episode_reward
    return rewards / n


r = np.ones(m2 + 1)
gamma = 0.9999
r[-1] = 0
inv = np.linalg.inv(np.identity(m2 + 1) - gamma * P)
v = np.dot(
    inv, np.dot(P, r)
)
v = np.round(v, 2)


def estimate_state_values(P, m2, threshold):
    v = np.zeros(m2 + 1)
    terminal_state = m2
    max_change = threshold
    while max_change >= threshold:
        max_change = 0
        for s in range(m2 + 1):
            v_new = 0
            print(v)
            for s_next in range(m2 + 1):
                r = 1 * (s_next != terminal_state)  # 1 for any except last row (absorption state)
                v_new += P[s, s_next] * (r + v[s_next])
            max_change = max(max_change, np.abs(v[s] - v_new))
            # print(max_change)
            v[s] = v_new
    return np.round(v, 2)


estimations = estimate_state_values(P, m2, 0.01)













































































































































































































































































