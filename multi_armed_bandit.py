import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cufflinks as cf
import plotly.offline

np.random.seed(5)


class GaussianBandit:

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def pull_lever(self):
        reward = np.random.normal(self.mean, self.std)
        return np.round(reward, 1)


class GaussianBanditGame:

    def __init__(self, bandits):
        self.bandits = bandits
        self.rewards, self.total_reward, self.n_played = [], 0, 0
        np.random.shuffle(self.bandits)

    def reset_game(self):
        self.rewards, self.total_reward, self.n_played = [], 0, 0

    def play(self, choice):
        reward = self.bandits[choice].pull_lever()
        self.rewards.append(reward)
        self.total_reward += reward
        self.n_played += 1
        return reward

    def user_play(self):
        self.reset_game()
        print('Game started, enter number outside %s to end the game' % np.arange(len(self.bandits)))
        while True:
            print('\n -- Round %d' % self.n_played)
            try:
                choice = int(input('Choose a machine from 0 to %d :' % (len(self.bandits) - 1)))
            except ValueError:
                break
            if choice in np.arange(len(self.bandits)):
                reward = self.play(choice)
                print('Machine %d gave reward %f' % (choice, reward))
                avg_reward = self.total_reward / self.n_played
                print('Your average reward so far is %f' % avg_reward)
            else:
                break
        print('Game has ended')
        if self.n_played > 0:
            print('Total reward is %f after %d round/s' % (self.total_reward, self.n_played))
            avg_reward = self.total_reward / self.n_played
            print('Average reward is %f' % avg_reward)


slotA = GaussianBandit(5, 3)
slotB = GaussianBandit(6, 2)
slotC = GaussianBandit(1, 5)
game = GaussianBanditGame([slotA, slotB, slotC])
# game.user_play()


class BernoulliBandit:

    def __init__(self, p):
        self.p = p

    def display_ad(self):
        reward = np.random.binomial(1, self.p)
        return reward


ads = [BernoulliBandit(i) for i in [0.004, 0.016, 0.020, 0.028, 0.031]]


#  EXPLORATION STRATEGIES

def A_B_n():

    n_test = 10_000
    Q = np.zeros(len(ads))  # Q, action values
    N = np.zeros(len(ads))  # N, total impressions
    total_reward = 0
    avg_rewards = []

    for i in range(n_test):
        ad = np.random.randint(len(ads))
        R = ads[ad].display_ad()
        N[ad] = N[ad] + 1
        Q[ad] = Q[ad] + (1 / N[ad]) * (R - Q[ad])
        total_reward = total_reward + R
        avg_rewards.append(total_reward / (i + 1))

    best_ad_index = np.argmax(Q)
    print(best_ad_index)

    plt.plot(avg_rewards)
    plt.xlabel('impressions')
    plt.ylabel('avg reward')
    plt.title('A/B/n test avg reward')
    plt.show()
    plt.clf()


#  E-GREEDY ACTIONS IMPROVEMENT

def epsilon_greedy(eps=0.1):

    n_prod = 100_000

    Q = np.zeros(len(ads))  # Q, action values
    N = np.zeros(len(ads))  # N, total impressions
    total_reward = 0
    avg_rewards = []

    for i in range(n_prod):
        if np.random.uniform(0, 1) <= eps or i == 0:
            ad = np.random.randint(len(ads))
        else:
            ad = np.argmax(Q)
        R = ads[ad].display_ad()
        N[ad] = N[ad] + 1
        Q[ad] = Q[ad] + (1 / N[ad]) * (R - Q[ad])
        total_reward = total_reward + R
        avg_rewards.append(total_reward / (i + 1))

    return np.argmax(Q), avg_rewards, total_reward


# epsilons = [0.1, 0.2, 0.5]
# col = ['r', 'b', 'g']
# for i in range(len(epsilons)):
#     eps = epsilon_greedy(epsilons[i])
#     plt.plot(eps[1], linewidth=0.5, c=col[i],
#              label='epsilon %.1f: avg.rew: %f, best action: %d'
#                    % (epsilons[i], eps[1][-1], int(eps[0])))
#
# plt.title('Epsilon greedy improvement')
# plt.xlabel('steps')
# plt.ylabel('avg reward')
# plt.tight_layout()
# plt.legend()
# plt.show()
# plt.clf()


#  UCB

def ucb(c=0.1):

    n_prod = 100_000
    n_ads = len(ads)
    ad_indices = np.arange(n_ads)
    Q = np.zeros(n_ads)
    N = np.zeros(n_ads)
    total_reward = 0
    avg_reward = []

    for t in range(n_prod):
        if any(N == 0):
            ad = np.random.choice(ad_indices[N == 0])
        else:
            uncertainty = np.sqrt(np.log(t+1) / N)
            ad = np.argmax(Q + c * uncertainty)
        R = ads[ad].display_ad()
        N[ad] = N[ad] + 1
        Q[ad] = Q[ad] + (1 / N[ad]) * (R - Q[ad])
        total_reward += R
        avg_reward.append(total_reward / (t+1))

    return np.argmax(Q), avg_reward, total_reward


# cs = [0.1, 1, 10]
# col = ['r', 'b', 'g']
# for i in range(len(cs)):
#     ucb_ = ucb(cs[i])
#     plt.plot(ucb_[1], linewidth=0.5, c=col[i],
#              label='ucb, c=%.1f: avg.rew: %f, best action: %d'
#                    % (cs[i], ucb_[1][-1], int(ucb_[0])))
# plt.title('Upper Confidence Bound')
# plt.xlabel('steps')
# plt.ylabel('avg reward')
# plt.tight_layout()
# plt.legend()
# plt.show()
# plt.clf()


#  Thompson Posterior Sampling

def thompson():

    n_prod = 100_000
    n_ads = len(ads)
    alphas = np.ones(n_ads)
    betas = np.ones(n_ads)
    total_reward = 0
    avg_reward = []

    for i in range(n_prod):
        theta_samples = [
            np.random.beta(alphas[k], betas[k]) for k in range(n_ads)
        ]
        ad = np.argmax(theta_samples)
        R = ads[ad].display_ad()
        alphas[ad] = alphas[ad] + R
        betas[ad] = betas[ad] + (1 - R)
        total_reward = total_reward + R
        avg_reward.append(total_reward / (i + 1))
        if i == n_prod - 1:
            best_K = np.argmax(alphas)

    return best_K, avg_reward, total_reward


t_ = thompson()
plt.plot(t_[1][1000:], linewidth=0.5, c='b',
             label='thompson, avg.rew: %f, best action: %d'
                   % (t_[1][-1], int(t_[0])))
plt.title('Thompson')
plt.xlabel('steps')
plt.ylabel('avg reward')
plt.ylim(ymin=0, ymax=0.05)
plt.tight_layout()
plt.legend()
plt.show()
plt.clf()









































































































































































































































































































































