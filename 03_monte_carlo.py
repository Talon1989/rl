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

policy = lambda state_, limit_=19: 0 if state_[0] > limit_ else 1

n_times = 100


def generate_episode(pol):
    episode_ = []
    state_ = env.reset()
    while True:
        action = pol(state_)
        next_state, reward, done, info = env.step(action)
        episode_.append([state_, action, reward])
        if done:
            break
        state_ = next_state
    return episode_


# episode_ = generate_episode(policy)


def every_visit_MC_state_values(n_iter=50_000):
    total_return_ = defaultdict(float)
    N_ = defaultdict(int)
    for i in range(n_iter):
        episode = generate_episode(policy)
        states, actions, rewards = zip(*episode)
        for t, state in enumerate(states):
            R = sum(rewards[t:])
            total_return_[state] = total_return_[state] + R
            N_[state] = N_[state] + 1
    return total_return_, N_


def first_visit_MC_state_values(n_iter=50_000):
    total_return_ = defaultdict(float)
    N_ = defaultdict(int)
    for i in range(n_iter):
        episode = generate_episode(policy)
        states, actions, rewards = zip(*episode)
        for t, state in enumerate(states):
            if state not in states[0: t]:
                R = sum(rewards[t:])
                total_return_[state] = total_return_[state] + R
                N_[state] = N_[state] + 1
    return total_return_, N_


total_return, N = first_visit_MC_state_values()

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


# total_return, N = every_visit_MC_state_values()
#
# total_return = pd.DataFrame(
#     total_return.items(), columns=['state', 'total_return']
# )
# N = pd.DataFrame(
#     N.items(), columns=['state', 'N']
# )
# df_2 = pd.merge(total_return, N, on='state')
# df_2['value'] = df_2['total_return'] / df_2['N']






























































































































































































































































































