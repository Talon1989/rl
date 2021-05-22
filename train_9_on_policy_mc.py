import numpy as np
import pandas as pd
from collections import defaultdict
import gym


env = gym.make('Blackjack-v0')
Q = defaultdict(float)
total_return = defaultdict(float)
N = defaultdict(int)
n_actions = env.action_space.n


def e_greedy(s, epsilon=0.1):
    if np.random.uniform(0, 1) < epsilon:
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
        next_state, reward, abs_state, _ = env.step(action)
        episode.append([state, action, reward])
        if abs_state:
            break
        state = next_state
    return episode


def monte_carlo_first_visit(n_iter=10_000):
    for i in range(n_iter):
        episode = create_episode()
        s_a_pairs = [(s, a) for s, a, _ in episode]
        rewards = [r for _, _, r in episode]
        for t, (s, a, _) in enumerate(episode):
            if (s, a) not in s_a_pairs[0:t]:
                total_return[(s, a)] += np.sum(rewards[t:])
                N[(s, a)] += 1
                Q[(s, a)] = total_return[(s, a)] / N[(s, a)]
    return pd.merge(
        pd.DataFrame(Q.items(), columns=['(s, a)', 'value']),
        pd.DataFrame(N.items(), columns=['(s, a)', 'n']),
        on='(s, a)'
    )


def monte_carlo_every_visit(n_iter=10_000):
    for i in range(n_iter):
        episode = create_episode()
        for t, (s, a, r) in enumerate(episode):
            total_return[(s, a)] += np.sum(np.array(episode, dtype=object)[t:, -1])
            N[(s, a)] += 1
            Q[(s, a)] = total_return[(s, a)] / N[(s, a)]
    return pd.merge(
        pd.DataFrame(Q.items(), columns=['(s, a)', 'value']),
        pd.DataFrame(N.items(), columns=['(s, a)', 'n']),
        on='(s, a)'
    )


# data = monte_carlo_every_visit()


env = gym.make('FrozenLake-v0')
n_actions = env.action_space.n
n_states = env.observation_space.n
P = env.P


def value_iteration(gamma=0.99):
    threshold = 1e-20
    state_table = np.zeros(n_states)
    while True:
        prev_state_table = state_table.copy()
        for s in range(n_states):
            q_values = [
                np.sum([
                    (r + gamma * prev_state_table[s_]) * p
                    for p, s_, r, _ in P[s][a]
                ]) for a in range(n_actions)
            ]
            state_table[s] = np.max(q_values)
        if np.sum(np.abs(prev_state_table - state_table)) < threshold:
            break
    policy = []
    for s in range(n_states):
        q_values = [
            np.sum([
                (r + gamma * prev_state_table[s_]) * p
                for p, s_, r, _ in P[s][a]
            ]) for a in range(n_actions)
        ]
        policy.append(np.argmax(q_values))
    return np.array(policy), state_table


def policy_iteration(gamma=0.99):
    threshold = 1e-20
    policy = np.random.randint(0, n_actions, n_states)
    while True:
        state_table = np.zeros(n_states)
        while True:
            prev_state_table = state_table.copy()
            for s in range(n_states):
                state_table[s] = np.sum([
                    (r + gamma * prev_state_table[s_]) * p
                    for p, s_, r, _ in P[s][policy[s]]
                ])
            if np.sum(np.abs(prev_state_table - state_table)) < threshold:
                break
        new_policy = []
        for s in range(n_states):
            q_values = [
                np.sum([
                    (r + gamma * prev_state_table[s_]) * p
                    for p, s_, r, _ in P[s][a]
                ]) for a in range(n_actions)
            ]
            new_policy.append(np.argmax(q_values))
        new_policy = np.array(new_policy)
        if (policy == new_policy).all():
            break
        policy = new_policy
    return policy, state_table


policy_, state_table_ = policy_iteration()













































































































































































































































































































































































