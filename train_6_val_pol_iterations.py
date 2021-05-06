import numpy as np
import gym


env_ = gym.make('FrozenLake-v0')
# env_.render()
n_states = env_.observation_space.n
n_actions = env_.action_space.n
P = env_.P


def value_iteration(threshold=1e-20, gamma=0.99):

    state_values = np.zeros(n_states)
    counter = 0

    while True:
        prev_state_values = state_values.copy()
        for s in range(n_states):
            q_values = [
                np.sum([
                    (r + gamma * prev_state_values[s_]) * p
                    for p, s_, r, _ in P[s][a]
                ]) for a in range(n_actions)
            ]
            state_values[s] = max(q_values)
        if np.sum(np.abs(prev_state_values - state_values)) < threshold:
            print('breaking iterations at iter:%d' % counter)
            break
        counter += 1

    best_policy = np.zeros(n_states)
    for s in range(n_states):
        q_values = [
            np.sum([
                (r + gamma * prev_state_values[s_]) * p
                for p, s_, r, _ in P[s][a]
            ]) for a in range(n_actions)
        ]
        best_policy[s] = np.argmax(q_values)

    return state_values, best_policy


# s, policy_ = value_iteration()


def policy_iteration(threshold=1e-20, gamma=0.99):

    policy = np.zeros(n_states)
    counter = 0

    while True:
        state_values = np.zeros(n_states)
        while True:
            prev_state_values = state_values.copy()
            for s in range(n_states):
                s_value = np.sum(
                    [(r_ + gamma * prev_state_values[s_]) * p for p, s_, r_, _ in P[s][policy[s]]]
                )
                state_values[s] = s_value
            if np.sum(np.abs(prev_state_values - state_values)) < threshold:
                break
        prev_policy = policy.copy()
        for s in range(n_states):
            q_values = [
                np.sum([
                    (r + gamma * prev_state_values[s_]) * p
                    for p, s_, r, _ in P[s][a]
                ]) for a in range(n_actions)
            ]
            policy[s] = np.argmax(q_values)
        if (prev_policy == policy).all():
            print('breaking iterations at iter:%d' % counter)
            break
        counter += 1

    return state_values, policy


# s, policy_ = policy_iteration()
































































































































































































































































































































