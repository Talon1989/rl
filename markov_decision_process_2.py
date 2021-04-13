import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import markov_decision_process


m = 3
m2 = m ** 2
P = np.zeros([m2 + 1, m2 + 1])
#  cols = from, rows = to
P[:m2, :m2] = markov_decision_process.get_P(3, 0.2, 0.3, 0.25, 0.25).T
for i in range(m2):
    P[m2, i] = P[i, i]
    P[i, i] = 0
P[m2, m2] = 1  # turning last col into absorbing state


#  assigning rewards to transitions
def average_rewards():
    n = 10 ** 3
    rewards = np.zeros(m2)
    for s in range(m2):
        for i in range(n):
            crashed = False
            s_next = s
            episode_reward = 0
            while not crashed:
                s_next = np.random.choice(m2 + 1, p=P[:, s_next])
                if s_next < m2:
                    episode_reward += 1
                else:  # absorbing state
                    crashed = True
            rewards[s] += episode_reward
    return rewards / n


# corners have lowest reward because of high probability of hitting walls
avg_rewards = average_rewards().reshape([3, 3])











































































































































































































































































































