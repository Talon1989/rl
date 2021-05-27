import numpy as np
import gym
import pandas as pd
from collections import defaultdict


env = gym.make('Blackjack-v0')
n_actions = env.action_space.n

Q = defaultdict(float)
total_return = defaultdict(float)
N = defaultdict(int)


def e_greedy(s, e=0.1):
    if np.random.uniform(0, 1) < e:
        action = np.random.randint(0, n_actions)
    else:
        action = np.argmax(
            [Q[(s, a)] for a in range(n_actions)]
        )
    return action


def create_episode():
    episode = []
    state = env.reset()
    while True:
        action = e_greedy(state)
        new_state, reward, absorption_state, _ = env.step(action)
        episode.append([state, action, reward])
        if absorption_state:
            break
        state = new_state
    return episode


def mc_e_greedy(n_iter=10_00):
    for i in range(n_iter):
        episode = create_episode()
        s_a_pairs = [(s, a) for s, a, _ in episode]
        rewards = [r for _, _, r in episode]
        for t, (s, a, _) in enumerate(episode):
            if (s, a) not in s_a_pairs[:t]:
                total_return[(s, a)] += np.sum(rewards[t:])
                N[(s, a)] += 1
                Q[(s, a)] = total_return[(s, a)] / N[(s, a)]
    return pd.merge(
        pd.DataFrame(Q.items(), columns=['(s, a)', 'value']),
        pd.DataFrame(N.items(), columns=['(s, a)', 'n'])
    )


import scipy.stats as stats
print(stats.norm(0, 1).cdf(0))
print(stats.binom(1, 0.8).pmf(1))



































































































































































































































































































































































































