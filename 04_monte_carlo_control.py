import random

import numpy as np
import pandas as pd
from collections import defaultdict
import gym

env = gym.make('Blackjack-v0')
env.seed(0)


def epsilon_greedy_policy(q, state, e=0.1):
    if random.uniform(0, 1) < e:
        return env.action_space.sample()
    else:
        # return max(
        #     list(range(env.action_space.n)),
        #     key=lambda a: q[(state, a)]
        # )
        return np.argmax(
            [q[(state, a)] for a in range(env.action_space.n)]
        )


def generate_episode(q):
    episode = []
    state = env.reset()
    while True:
        action = epsilon_greedy_policy(q, state)
        next_state, reward, done, _ = env.step(action)
        episode.append([state, action, reward])
        if done:
            break
        state = next_state
    return episode


#  ON-POLICY
def control_on_policy(n_iter=100_000):

    Q_ = defaultdict(float)  # q-values (action-values)
    total_return_ = defaultdict(float)  # total return of q(s,a)
    N_ = defaultdict(int)  # number of times (s, a) pair is visited

    for i in range(n_iter):
        episode = generate_episode(Q_)
        state_action_pairs = [(s, a) for s, a, _ in episode]
        rewards = [r for _, _, r in episode]
        for t, (state, action, _) in enumerate(episode):
            if not (state, action) in state_action_pairs[0:t]:
                R = sum(rewards[t:])
                total_return_[(state, action)] += R
                N_[(state, action)] += 1
                Q_[(state, action)] = total_return_[(state, action)] / N_[(state, action)]

    return pd.DataFrame(Q_.items(), columns=['(s, a) pair', 'value'])


df = control_on_policy()

















































































































































































































































































































