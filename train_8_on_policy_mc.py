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
    if np.random.uniform(0, 1) < e or len(Q) == 0:
        action = np.random.randint(0, n_actions)
    else:
        action = np.argmax(
            [Q[(s, a)] for a in range(n_actions)]
        )
    return action


def generate_episode():
    episode_ = []
    state = env.reset()
    while True:
        action = e_greedy(state)
        next_state, reward, done, _ = env.step(action)
        episode_.append([state, action, reward])
        if done:
            break
        state = next_state
    return episode_


env.seed(0)
n_iter = 100_000
for i in range(n_iter):
    episode = generate_episode()
    s_a_pairs = [(s, a) for s, a, _ in episode]
    rewards = [r for _, _, r in episode]
    for t, (s, a, _) in enumerate(episode):
        if not (s, a) in s_a_pairs[0:t]:  # do not recount same (s, a) in same episode
            R = sum(rewards[t:])
            total_return[(s, a)] += R
            N[(s, a)] += 1
            Q[(s, a)] = total_return[(s, a)] / N[(s, a)]

df = pd.DataFrame(Q.items(), columns=['(s, a)', 'value'])
nn = pd.DataFrame(N.items(), columns=['(s, a)', 'n'])
a = pd.merge(df, nn, on='(s, a)')


























































































































































































































































































































