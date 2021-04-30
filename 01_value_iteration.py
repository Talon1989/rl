import numpy as np
import gym


env_ = gym.make('FrozenLake-v0')
env_.render()


def value_iteration(env, n_iter=1000):
    np.random.seed(0)
    threshold = 1e-20  # threshold to check convergence
    gamma = 0.99
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    value_table = np.zeros(n_states)
    # value_table = np.random.normal(
    #     loc=0, scale=0.1, size=n_states
    # )
    for i in range(n_iter):
        updated_value_table = np.copy(value_table)
        for s in range(n_states):
            q_values = [
                sum([(r + gamma*updated_value_table[s_]) * prob
                     for prob, s_, r, _ in env.P[s][a]])
                for a in range(n_actions)
            ]
            value_table[s] = max(q_values)
        if np.sum(np.abs(updated_value_table - value_table)) <= threshold:
            print('breaking iterations at iter n: %d' % i)
            break
    return value_table


def extract_policy(env, value_table):
    gamma = 0.99
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = np.zeros(n_states)
    for s in range(n_states):
        q_values = [
            sum([(r + gamma*value_table[s_]) * prob
                 for prob, s_, r, _ in env.P[s][a]])
            for a in range(n_actions)
        ]
        policy[s] = np.argmax(q_values)
    return policy


table = value_iteration(env_)
pol = extract_policy(env_, table)






















































































































































































































































































































































































