import numpy as np
import gym


env = gym.make('FrozenLake-v0')
n_states = env.observation_space.n
n_actions = env.action_space.n  # l, d, r ,u
real_target = np.array([0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0])


def q_learning(n_episodes=1_000):

    q = np.zeros([n_states, n_actions])
    epsilon = 1
    epsilon_decay = 99/100
    epsilon_min = 1/100
    alpha = 1/10
    gamma = 99/100

    def e_greedy(s):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(0, n_actions)
        else:
            return np.argmax(q[s])

    for e in range(n_episodes):
        s = env.reset()
        while True:
            a = e_greedy(s)
            s_, r, done, _ = env.step(a)
            q[s, a] = q[s, a] + alpha * (r + gamma * np.max(q[s_]) - q[s, a])
            if done:
                break
            s = s_
        if epsilon > epsilon_min:
            epsilon = epsilon * epsilon_decay

    policy = np.zeros(n_states)
    for s in range(n_states):
        policy[s] = np.argmax(q[s])
    return policy


# pol = q_learning(1_000)
# print(np.sum(pol != real_target))


def value_iteration():
    P = env.P
    threshold = 1e-16
    gamma = 99/100
    v = np.zeros(n_states)
    counter = 0
    while True:
        old_v = v.copy()
        for s in range(n_states):
            v[s] = np.max([
                np.sum([
                    (r + gamma * old_v[s_]) * p for p, s_, r, _ in P[s][a]
                ]) for a in range(n_actions)
            ])
        if np.sum(np.abs(old_v - v)) < threshold:
            print('breaking at %d iteration' % (counter + 1))
            break
        counter += 1
    policy = np.zeros(n_states)
    for s in range(n_states):
        policy[s] = np.argmax([
            np.sum([
                (r + gamma * old_v[s_]) * p for p, s_, r, _ in P[s][a]
            ]) for a in range(n_actions)
        ])
    return policy


pol = value_iteration()
print(np.sum(pol != real_target))


















































































































































































































































































































