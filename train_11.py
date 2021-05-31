import numpy as np
import gym
import pandas as pd
import time


np.random.seed(1)
env = gym.make('FrozenLake-v0')  # l, d, r ,u
# env = gym.make('Taxi-v3')  # s, n, e, w, pick, drop
env.seed(1)
n_actions = env.action_space.n
n_states = env.observation_space.n
q = np.zeros([n_states, n_actions])


def epsilon_greedy(s, epsilon=0.3):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, n_actions)
    else:
        return np.argmax(s)


def simulation():
    s = env.reset()
    while True:
        env.render()
        a = np.argmax(q[s])
        s, r, done, _ = env.step(a)
        print(r)
        time.sleep(0.5)
        if done:
            break


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


def q_learning(alpha=0.01, gamma=0.99, n_episodes=10000, epsilon_decay_rate=0.0005):
    epsilon = 0.3
    for episode in range(n_episodes):
        s = env.reset()
        a = np.argmax(q[s])
        while True:
            s_, r, done, _ = env.step(action=a)
            a_ = np.argmax(q[s_])
            q[s][a] = q[s][a] + alpha * (r + (gamma * q[s_][a_]) - q[s][a])
            if done:
                break
            if epsilon > 0.01:
                pass
                # epsilon -= epsilon_decay_rate
            s = s_
            a = epsilon_greedy(s, epsilon)


# rewards = sarsa()
q_learning(alpha=0.6, n_episodes=50_000)


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


# df = get_a_s_pair_dataframe()
pp = get_policy()


def test_taxi():
    env = gym.make('Taxi-v3')
    env.reset()
    while True:
        a = int(input('input action'))
        if a in np.arange(n_actions):
            _, _, done, _ = env.step(a)
            env.render()
            if done:
                break
        else:
            print('please enter input Z between 0 and 5')


def value_iteration():
    gamma = 0.99
    state_values = np.zeros(n_states)
    threshold = 1e-16
    P = env.P
    counter = 0
    while True:
        state_values_copy = state_values.copy()
        for s in range(n_states):
            state_values[s] = np.max([
                np.sum([
                    (r + gamma * state_values_copy[s_]) * p
                    for p, s_, r, _ in P[s][a]
                ])
                for a in range(n_actions)
            ])
        counter += 1
        if np.sum(np.abs(state_values_copy - state_values)) < threshold:
            print(counter)
            break
    policy = []
    for s in range(n_states):
        q_values = [
            np.sum([
                (r + gamma * state_values[s_]) * p
                for p, s_, r, _ in P[s][a]
            ])
            for a in range(n_actions)
        ]
        policy.append(np.argmax(q_values))
    return np.array(policy)


def value_iteration_simulation(pol):
    s = env.reset()
    while True:
        env.render()
        a = pol[s]
        s, r, done, _ = env.step(a)
        time.sleep(0.5)
        if done:
            break


# p = value_iteration()
# value_iteration_simulation(p)





































































































































































































































