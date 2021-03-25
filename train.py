import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


np.random.seed(1)


class action:

    def __init__(self, p):
        self.p = p

    def get_action(self):
        return np.random.binomial(1, self.p)


probas = [0.1, 0.4, 0.2]
actions = [
    action(i) for i in probas
]
n_actions = len(actions)
n_iter = 10_000


def e_greedy(e):

    Q, N = np.zeros(n_actions), np.zeros(n_actions)
    avg_reward = []
    sum_rewards = 0
    for i in range(n_iter):
        if np.random.uniform(0, 1) > e and i > 0:
            a = np.argmax(Q)
        else:
            a = np.random.randint(n_actions)
        N[a] += 1
        R = actions[a].get_action()
        Q[a] = Q[a] + (1/N[a]) * (R - Q[a])
        sum_rewards += R
        avg_reward.append(sum_rewards / (i+1))
    return np.argmax(Q), avg_reward

# es = [0.1, 0.3, 0.5, 1]
# lines = ['-', '--', '-.', '-']
# colors = ['b', 'g', 'r', 'y']
# for i in range(len(es)):
#     plt.plot(e_greedy(es[i])[1][10:], linestyle=lines[i], c=colors[i], label='e=%.1f' % es[i])
# plt.title('e-greedy')
# plt.xlabel('iterations')
# plt.ylabel('avg values')
# plt.legend()
# plt.show()
# plt.clf()


def ucb(c):

    Q, N = np.zeros(n_actions), np.zeros(n_actions)
    avg_reward = []
    sum_rewards = 0
    for i in range(n_iter):
        if any(N == 0):
            a = np.random.choice(np.arange(n_actions)[N == 0])
        else:
            uncertainty = c * np.sqrt(np.log(i+1) / N)
            a = np.argmax(Q + uncertainty)
        N[a] += 1
        R = actions[a].get_action()
        Q[a] = Q[a] + (1/N[a]) * (R - Q[a])
        sum_rewards += R
        avg_reward.append(sum_rewards / (i+1))
    return np.argmax(Q), avg_reward

# cs = [0.01, 0.1, 1, 10]
# lines = ['-', '--', '-.', '-']
# colors = ['b', 'g', 'r', 'y']
# for i in range(len(cs)):
#     plt.plot(ucb(cs[i])[1][10:], linestyle=lines[i], c=colors[i], label='e=%.2f' % cs[i])
# plt.title('ucb')
# plt.xlabel('iterations')
# plt.ylabel('avg values')
# plt.legend()
# plt.show()
# plt.clf()


def thompson():

    alphas, betas = np.ones(n_actions), np.ones(n_actions)
    sum_rewards = 0
    avg_rewards = []
    for i in range(n_iter):
        a = np.argmax(np.random.beta(alphas, betas))
        R = actions[int(a)].get_action()
        alphas[a] += R
        betas[a] += (1 - R)
        sum_rewards += R
        avg_rewards.append(sum_rewards / (i+1))
    return np.argmax(alphas), avg_rewards


plt.plot(thompson()[1][10:], c='b', linewidth=0.5)
plt.title('thompson')
plt.xlabel('iterations')
plt.ylabel('avg values')
plt.show()
plt.clf()
























































































































































































































