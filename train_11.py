import numpy as np
import gym
import pandas as pd


env = gym.make('FrozenLake-v0')
n_actions = env.action_space.n  # l, d, r ,u
n_states = env.observation_space.n
q = np.zeros([n_states, n_actions])


def epsilon_greedy(s, epsilon=0.3):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, n_actions)
    else:
        return np.argmax(s)


def sarsa(alpha=0.01, gamma=0.95, n_episodes=10000):
    tot_reward = []
    for episode in range(n_episodes):
        s = env.reset()
        a = epsilon_greedy(s)
        reward = 0
        while True:
            s_, r, done, _ = env.step(action=a)
            a_ = epsilon_greedy(s_)
            q[s][a] = q[s][a] + alpha * (r + gamma * q[s_][a_] - q[s][a])
            reward += q[s][a]
            if done:
                tot_reward.append(reward)
                break
            s = s_
            a = a_
    return np.array(tot_reward)


def q_learning(alpha=0.01, gamma=0.95, n_episodes=10000):
    for episode in range(n_episodes):
        s = env.reset()
        a = np.argmax(q[s])
        while True:
            s_, r, done, _ = env.step(action=a)
            a_ = np.argmax(q[s_])
            q[s][a] = q[s][a] + alpha * (r + gamma * q[s_][a_] - q[s][a])
            if done:
                break
            s = s_
            a = epsilon_greedy(s)


# rewards = sarsa()
q_learning()


def get_a_s_pair_dataframe():
    pair_names = []
    form_data = q.reshape([-1, 1])
    for s in range(n_states):
        for a in range(n_actions):
            pair_names.append('(%d, %d)' % (s, a))
    pair_names = np.array(pair_names).reshape([-1, 1])
    frame = np.hstack([pair_names, np.round(form_data, 4)])
    return pd.DataFrame({'(s, a)': frame[:, 0], 'value': frame[:, 1]})


def get_policy():
    policy = []
    for s in range(n_states):
        policy.append(np.argmax(q[s]))
    return np.array(policy)


df = get_a_s_pair_dataframe()
pp = get_policy()













































































































































































































































