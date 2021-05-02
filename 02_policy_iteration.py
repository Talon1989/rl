import numpy as np
import gym


env_ = gym.make('FrozenLake-v0')
# env_.render()
n_states = env_.observation_space.n
n_actions = env_.action_space.n
P = env_.P


# initial policy is initialized as all first action
policy_ = np.zeros(n_states)


def compute_state_value_function(policy, n_iter=1000):
    threshold = 1e-20
    gamma = 0.99
    value_table = np.zeros(n_states)
    for i in range(n_iter):
        update_value_table = value_table.copy()
        for s in range(n_states):
            a = policy[s]
            v_values = sum(
                (r + gamma * update_value_table[s_]) * p
                for p, s_, r, _ in P[s][a]
            )
            value_table[s] = v_values
        if sum(np.abs(update_value_table - value_table)) <= threshold:
            print('breaking iteration at iter %d' % i)
            break
    return value_table


def extract_policy(state_value_table):
    gamma = 0.99
    policies = []
    for s in range(n_states):
        q_values = [
            sum((r + gamma * state_value_table[s_]) * p
                for p, s_, r, _ in P[s][a])
            for a in range(n_actions)
        ]
        policies.append(np.argmax(q_values))
    return np.array(policies)


# state_value_table_ = compute_state_value_function(policy_)
# new_policy = extract_policy(state_value_table_)


def compute_best_policy(policy):
    old_policy = np.full(shape=n_states, fill_value=-1)
    new_policy = policy
    counter = 0
    while not (old_policy == new_policy).all():
        old_policy = new_policy
        state_value_table_ = compute_state_value_function(new_policy)
        new_policy = extract_policy(state_value_table_)
        counter += 1
    print('recalculations to best policy: %d' % counter)
    return new_policy


best_policy = compute_best_policy(policy_)



































































































































































































































