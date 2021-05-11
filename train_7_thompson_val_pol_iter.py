import numpy as np
import gym


def thompson(n_iter=1000):
    action_probas = np.array([0.2, 0.6, 0.4])
    beta_params = np.ones([action_probas.shape[0], 2])
    for _ in range(n_iter):
        action_predict = np.argmax(np.random.beta(beta_params[:, 0], beta_params[:, 1]))
        reward = np.random.binomial(1, action_probas[action_predict])
        beta_params[action_predict, 0] += reward
        beta_params[action_predict, 1] += (1 - reward)


###############################################################


env = gym.make('FrozenLake-v0')

P = env.P
n_actions = env.action_space.n
n_states = env.observation_space.n


def value_iteration(graph=False):
    """
    :return:  tuple(v*, policy*)
    """

    threshold = 1e-20
    gamma = 0.99
    state_values = np.zeros(n_states)
    counter = 0
    error = []

    while True:

        old_state_values = state_values.copy()
        for s in range(n_states):
            q_func = [
                np.sum([
                    (r + gamma * old_state_values[s_]) * p for p, s_, r, _ in P[s][a]
                ]) for a in range(n_actions)
            ]
            state_values[s] = np.max(q_func)

        if np.sum(np.abs(old_state_values - state_values)) <= threshold:
            policy = np.zeros(n_states)
            for s in range(n_states):
                q_func = [
                    np.sum([
                        (r + gamma * old_state_values[s_]) * p for p, s_, r, _ in P[s][a]
                    ]) for a in range(n_actions)
                ]
                policy[s] = np.argmax(q_func)
            break

        if graph:
            error.append(np.sum(np.abs(old_state_values - state_values)))
        counter += 1
    print(counter)

    if graph:
        return state_values, policy, error
    return state_values, policy


def policy_iteration():
    """
    :return:  tuple(v*, policy*)
    """

    threshold = 1e-20
    gamma = 0.99
    policy = np.zeros(n_states)
    counter = 0
    while True:
        state_values = np.zeros(n_states)
        while True:
            state_values_old = state_values.copy()
            for s in range(n_states):
                state_values[s] = np.sum([
                    (r + gamma * state_values_old[s_]) * p
                    for p, s_, r, _ in P[s][policy[s]]
                ])
            if np.sum(np.abs(state_values_old - state_values)) <= threshold:
                break
        old_policy = policy.copy()
        for s in range(n_states):
            q_func = [
                np.sum([
                    (r + gamma * state_values[s_]) * p
                    for p, s_, r, _ in P[s][a]
                ]) for a in range(n_actions)
            ]
            policy[s] = np.argmax(q_func)
        if (old_policy == policy).all():
            break
        counter += 1
    print(counter)
    return state_values, policy


state_star, policy_star = value_iteration()
state_star_2, policy_star_2 = policy_iteration()















































































































































































































































