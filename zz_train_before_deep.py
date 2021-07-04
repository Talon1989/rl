import numpy as np
import gym
import pandas as pd


#  value iter / mc / q-learning / softmax / ucb / thompson


env = gym.make('FrozenLake-v0')
n_actions = env.action_space.n  # l, d, r ,u
n_states = env.observation_space.n
target = np.array([0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0])


def val_iter():
    P = gym.make('FrozenLake-v0').P
    v = np.zeros(n_states)
    threshold = 1e-16
    gamma = 0.99
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
    policy = np.zeros(n_states)
    for s in range(n_states):
        policy[s] = np.argmax([
            np.sum([
                (r + gamma * v[s_]) * p
                for p, s_, r, _ in P[s][a]
            ])
            for a in range(n_actions)
        ])
    return policy


policy = val_iter()


def mc(n_iter=10_000):

    from collections import defaultdict
    env = gym.make('Blackjack-v0')
    n_actions = env.action_space.n
    q = defaultdict(float)
    total_rewards = defaultdict(float)
    n = defaultdict(int)

    def e_greedy(s, e=0.1):
        if np.random.uniform(0, 1) < e:
            return np.random.randint(0, n_actions)
        else:
            return np.argmax([q[(s, a)] for a in range(n_actions)])

    def play_episode():
        s = env.reset()
        episode = []
        while True:
            action = e_greedy(s)
            s_, r, done, _ = env.step(action)
            episode.append([s, action, r])
            if done:
                break
        return episode

    for i in range(n_iter):
        episode = play_episode()
        s_a = [(s, a) for s, a, r in episode]
        rewards = [r for _, _, r in episode]
        for t, (s, a, r) in enumerate(episode):
            # if (s, a) not in s_a[:t]:
            total_rewards[(s, a)] += np.sum(rewards[t:])
            n[(s, a)] += 1
            q[(s, a)] = total_rewards[(s, a)] / n[(s, a)]

    return pd.merge(
        pd.DataFrame(q.items(), columns=['(s, a)', 'value']),
        pd.DataFrame(n.items(), columns=['(s, a)', 'n']),
        on='(s, a)'
    )


# mc = mc()


def q_learning(n_iter=10_000, alpha=0.1, gamma=0.99):

    q = np.zeros([n_states, n_actions])

    def e_greedy(s, e=0.1):
        if np.random.uniform(0, 1) < e or np.sum(np.abs(q[s])) == 0:
            return np.random.randint(0, n_actions)
        else:
            return np.argmax(q[s])

    for i in range(n_iter):
        s = env.reset()
        while True:
            a = e_greedy(s)
            s_, r, done, _ = env.step(a)
            q[s, a] = q[s, a] + alpha * (r + gamma * np.max(q[s_]) - q[s, a])
            if done:
                break
            s = s_
    return q


# q = q_learning()


def mab():

    from data import bandit_bruteforce
    env = bandit_bruteforce.BanditTwoArmedHighLowFixed()
    n_actions = env.action_space.n
    count = np.zeros(n_actions)
    sum_rewards = np.zeros(n_actions)
    q = np.zeros(n_actions)
    n_rounds = 10_000

    def e_greedy(e=0.1):
        if np.random.uniform(0, 1) < e:
            return np.random.randint(0, n_actions)
        else:
            return np.argmax(q)

    def softmax(i):
        denom = np.sum([np.exp(q[a] * (i/n_rounds)) for a in range(n_actions)])
        probas = [np.exp(q[a] * (i/n_rounds)) / denom for a in range(n_actions)]
        return np.random.choice(n_actions, p=probas)

    def UCB(i):
        ucb = np.zeros(n_actions)
        if i < n_actions:
            return i
        else:
            return np.argmax([
                q[a] + (2*np.log(i) / count[a]) for a in range(n_actions)
            ])

    def thompson(n_iter):
        betas = np.ones([n_actions, 2])
        for i in range(n_iter):
            action = np.argmax(
                np.random.beta(betas[:, 0], betas[:, 1])
            )
            _, r, _, _ = env.step(action)
            betas[action, 0] += r
            betas[action, 1] += 1 - r
        return betas[:, 0] / (betas[:, 0] + betas[:, 1])

    for i in range(n_rounds):
        s = env.reset()
        action = softmax(i)
        _, r, _, _ = env.step(action)
        sum_rewards[action] += r
        count[action] += 1
        q[action] = sum_rewards[action] / count[action]
    return q


q = mab()
























































































































































































































































































