import numpy as np
import pandas as pd
import gym
from collections import defaultdict
import matplotlib.pyplot as plt
import time


np.random.seed(1)
target = [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]
env = gym.make('FrozenLake-v0')  # l, d, r ,u
n_states = env.observation_space.n
n_actions = env.action_space.n


#  KNOW ENVIRONMENTS  #######################################


def policy_iteration(gamma=0.99, threshold=1e-16):
    P = env.P
    policy = np.zeros(n_states)
    while True:
        v = np.zeros(n_states)
        while True:
            old_v = v.copy()
            for s in range(n_states):
                v[s] = np.sum([
                    (r + gamma * old_v[s_]) * p
                    for p, s_, r, _ in P[s][policy[s]]
                ])
            if np.sum(np.abs(old_v - v)) < threshold:
                break
        old_policy = policy.copy()
        for s in range(n_states):
            policy[s] = np.argmax([
                np.sum([
                    (r + gamma * v[s_]) * p
                    for p, s_, r, _ in P[s][a]
                ]) for a in range(n_actions)
            ])
        if (old_policy == policy).all():
            break
    return policy


def value_iteration(gamma=0.99, threshold=1e-16):
    P = env.P
    v = np.zeros(n_states)
    while True:
        old_v = v.copy()
        for s in range(n_states):
            v[s] = np.max([
                np.sum([
                    (r + gamma * old_v[s_]) * p
                    for p, s_, r, _ in P[s][a]
                ])
                for a in range(n_actions)
            ])
        if np.sum(np.abs(old_v - v)) < threshold:
            break
    policy = np.empty(n_states)
    for s in range(n_states):
        policy[s] = np.argmax([
            np.sum([
                (r + gamma * v[s_]) * p
                for p, s_, r, _ in P[s][a]
            ])
            for a in range(n_actions)
        ])
    return policy


# pol = policy_iteration()
# pol = value_iteration()#
# print(
#     (pol == target).all()
# )


#  MONTE CARLO  #######################################


def mc(n_iter=10_000):

    env = gym.make('Blackjack-v0')
    n_actions = env.action_space.n
    q = defaultdict(float)
    total_rewards = defaultdict(float)
    n = defaultdict(int)

    def epsilon_greedy(s, epsilon=0.1):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(0, n_actions)
        return np.argmax([q[(s, a)] for a in range(n_actions)])

    def play_episode():
        episode = []
        s = env.reset()
        while True:
            a = epsilon_greedy(s)
            s_, r, done, _ = env.step(a)
            episode.append([s, a, r])
            if done:
                break
            s = s_
        return episode

    for i in range(n_iter):
        episode = play_episode()
        s_a_pairs = [(s, a) for s, a, _ in episode]
        rewards = [r for _, _, r in episode]
        for t, (s, a, _) in enumerate(episode):
            if (s, a) not in s_a_pairs[:t]:
                total_rewards[(s, a)] += np.sum(rewards[t:])
                n[(s, a)] += 1
                q[(s, a)] = total_rewards[(s, a)] / n[(s, a)]
    return pd.merge(
        pd.DataFrame(q.items(), columns=['(s, a)', 'value']),
        pd.DataFrame(n.items(), columns=['(s, a)', 'n']),
        on='(s, a)'
    )


# blackjack = mc()


#  Q-LEARNING  #######################################


q = np.zeros([n_states, n_actions])


def aug_epsilon_greedy(s, epsilon=0.1, augmented=True):
    if augmented and (np.random.uniform(0, 1) < epsilon or np.sum(np.abs(q[s]))) == 0:
        return np.random.randint(0, n_actions)
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, n_actions)
    return np.argmax(q[s])


def q_learning(alpha=0.3, gamma=0.99, n_episodes=50_000, aug_epsilon=True):
    for episode in range(n_episodes):
        s = env.reset()
        while True:
            a = aug_epsilon_greedy(s, aug_epsilon)
            s_, r, done, _ = env.step(a)
            q[s, a] = q[s, a] + alpha * (r + gamma * q[s_, np.argmax(q[s_])] - q[s, a])
            if done:
                break
            s = s_
    policy = np.empty(n_states)
    for s in range(n_states):
        policy[s] = np.argmax(q[s])
    return policy


pol = q_learning()
print(
    n_states - np.sum(pol == target)  # lower value, better result
)

#  TESTING

# aug_counter = 0
# cumulative_time_aug = 0
# for i in range(10):
#     begin_time = time.time()*1000
#     aug_counter += n_states - np.sum(q_learning(n_episodes=50_000) - target)
#     t = time.time()*1000 - begin_time
#     print('aug iteration %d took %f ms' % (i, t))
#     q = np.zeros([n_states, n_actions])
#     cumulative_time_aug += t
# print()
#
# vanilla_counter = 0
# cumulative_time_van = 0
# for i in range(10):
#     begin_time = time.time()*1000
#     vanilla_counter += n_states - np.sum(q_learning(n_episodes=50_000, aug_epsilon=False) - target)
#     t = time.time()*1000 - begin_time
#     print('van iteration %d took %f ms' % (i, t))
#     q = np.zeros([n_states, n_actions])
#     cumulative_time_van += t
# print()
#
# print('Cumulative time aug: %f\nCumulative time van: %f' % (cumulative_time_aug, cumulative_time_van))
# print('Lower is better: \nAugmented epsilon: %d\nVanilla epsilon: %d' % (aug_counter, vanilla_counter))



















































































































































































































































































