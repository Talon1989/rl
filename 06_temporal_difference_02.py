import numpy as np
import gym
import pandas as pd


env = gym.make('FrozenLake-v0')
n_actions = env.action_space.n  # l, d, r ,u
n_states = env.observation_space.n
q = {}
for s in range(n_states):
    for a in range(n_actions):
        q[(s, a)] = 0.


def e_greedy(s, epsilon=0.1):
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(0, n_actions)
    else:
        action = np.argmax([q[(s, a)] for a in range(n_actions)])
    return action


def sarsa():
    alpha, gamma = 0.85, 0.9
    n_episodes = 50_000
    for i in range(n_episodes):
        s = env.reset()
        a = e_greedy(s)
        while True:
            s_, r, end, _ = env.step(a)
            a_ = e_greedy(s_)
            q[(s, a)] = q[(s, a)] + alpha * (r + gamma * q[(s_, a_)] - q[(s, a)])
            s, a = s_, a_
            if end:
                break
    # policy = []
    # for s in range(n_states):
    #     policy.append(
    #         np.argmax([
    #             q[(s, a)] for a in range(n_actions)
    #         ])
    #     )
    # return np.array(policy)
    return pd.DataFrame(q.items(), columns=['(s, a)', 'value'])


def q_learning():
    alpha, gamma = 0.85, 0.95
    n_episodes = 500_000
    for i in range(n_episodes):
        s = env.reset()
        while True:
            a = e_greedy(s)  # e-greedy policy
            s_, r, end, _ = env.step(a)
            a_ = np.argmax(  # greedy policy
                [q[(s_, a__)] for a__ in range(n_actions)]
            )
            s_, a_ = int(s_), int(a_)  # to deal with pycharm warnings
            q[(s, a)] = q[(s, a)] + alpha * (r + gamma * q[(s_, a_)] - q[(s, a)])
            s = s_
            if end:
                break
    policy = []
    for s in range(n_states):
        policy.append(
            np.argmax([
                q[(s, a)] for a in range(n_actions)
            ])
        )
    return np.array(policy)
    # return pd.DataFrame(q.items(), columns=['(s, a)', 'value'])


# data = sarsa()
data = q_learning()


# d = pd.DataFrame({'a': a, 'b': b})




































































































































































































































































































































