import numpy as np
import pandas as pd
from collections import defaultdict
import gym

env = gym.make('Blackjack-v0')
env.seed(0)

# (value of our card, value dealer's card, usable ace)
print(env.reset())

print(env.observation_space)
print(env.action_space)  # stand=0, hit=1
print()

policy = lambda state, limit=19: 0 if state[0] > limit else 1

n_times = 100


def generate_episode(pol):
    episode_ = []
    state = env.reset()
    while True:
        action = pol(state)
        next_state, reward, done, info = env.step(action)
        episode_.append((state, action, reward))
        if done:
            break
        state = next_state
    return episode_


# episode_ = generate_episode(policy)

total_return = defaultdict(float)
N = defaultdict(int)

n_iter = 50_000
for i in range(n_iter):
    episode = generate_episode(policy)
    states, actions, rewards = zip(*episode)
    for t, state in enumerate(states):
        R = sum(rewards[t:])
        total_return[state] = total_return[state] + R
        N[state] = N[state] + 1

total_return = pd.DataFrame(
    total_return.items(), columns=['state', 'total_return']
)
N = pd.DataFrame(
    N.items(), columns=['state', 'N']
)
df = pd.merge(total_return, N, on='state')
df['value'] = df['total_return'] / df['N']

print(  # clearly good one
    df[df['state'] == (21, 9, False)]['value'].values
)
print(  # bad one
    df[df['state'] == (5, 8, False)]['value'].values
)


















































































































































































































































































