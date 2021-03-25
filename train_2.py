import numpy as np
import matplotlib.pyplot as plt


def dice_proba(n_iter):
    clt_proba = []
    results = 0
    for i in range(n_iter):
        if np.random.choice([1, 2, 3, 4, 5, 6]) == 1:
            results += 1
        clt_proba.append(
            results / (i + 1)
        )
    return clt_proba


# clt = dice_proba(100000)
# plt.plot(clt, c='b')
# plt.title(clt[-1])
# plt.show()
# plt.clf()


arr = [0, 1, 0, 0, 1, 5, 3, 2, 2, 1, 0, 5, 1]
dic = {}
for num in arr:
    try:
        dic[num] += 1
    except KeyError:
        dic[num] = 1


# a = np.random.binomial(1, 0.21)
# b = np.random.binomial(1, 0.24)
# c = np.random.binomial(1, 0.15)


np.random.seed(1)


p_ = [0.21, 0.24, 0.30, 0.25, 0.31]


def thompson(p, n_iter=1000):
    total_reward = 0
    average_rewards = []
    betas = [np.ones(2) for _ in range(len(p))]
    for i in range(n_iter):
        action = int(np.argmax([np.random.beta(a, b) for a, b in betas]))
        R = np.random.binomial(1, p[action])
        betas[action][0] = betas[action][0] + R
        betas[action][1] = betas[action][1] + (1 - R)
        total_reward += R
        average_rewards.append(total_reward / (i+1))
    return average_rewards, np.argmax(np.array(betas)[:, 0])


def e_greedy(p, e, n_iter=1000):
    total_reward = 0
    average_rewards = []
    Q = np.zeros(len(p))
    N = np.zeros(len(p))
    for i in range(n_iter):
        if np.random.uniform(0, 1) > e:
            action = np.argmax(Q)
        else:
            action = np.random.randint(len(p))
        R = np.random.binomial(1, p[action])
        N[action] += 1
        Q[action] = Q[action] + (1/N[action]) * (R - Q[action])
        total_reward += R
        average_rewards.append(total_reward / (i+1))
    return average_rewards, np.argmax(Q)


rewards_e = e_greedy(p_, 0.1, 4000)
rewards_t = thompson(p_, 4000)
plt.plot(rewards_e[0], label='e_greedy, best action: %d' % rewards_e[1], linewidth=1, linestyle='-')
plt.plot(rewards_t[0], label='thompson, best action: %d' % rewards_t[1], linewidth=1, linestyle='--')
plt.legend(loc='best')
plt.ylabel('reward')
plt.xlabel('iterations')
plt.show()
plt.clf()














































































































































































































