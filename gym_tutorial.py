import numpy as np
import gym


# env_2 = gym.make('SpaceInvaders-v0')
# print(env_2.observation_space)


env = gym.make('FrozenLake-v0')

#  we can get a random sample using env.observation_space.sample() or env.action_space.sample()
print(env.observation_space)
print(env.action_space)

print()
# env.P[state][action], actions=[0=left, 1=down, 2=right, 3=up], if out of bounds go back to same state
print(env.P[3][1])  # returns (trans proba, next state, reward, absorbing state?)

state = env.reset()

#  action selection

env.seed(6)

env.render()
# down command = probas: .3 left, .3 down, .3right
iter_1_log = env.step(1)  # returns tuple(selected state, reward, absorbing state?, proba)
env.render()
env.reset()


#  GENERATING AN EPISODE

def episode_generation():
    n_iterations = 20
    cumulative_reward, total_actions = 0, 0
    for iter in range(n_iterations):
        print(iter)
        random_action = env.action_space.sample()
        next_state, reward, absorbing_state, info = env.step(random_action)
        print('Iteration %d' % iter)
        cumulative_reward += reward
        total_actions += 1
        env.render()
        if absorbing_state:
            break
    env.reset()
    return cumulative_reward, total_actions


def cart_pole():
    env = gym.make('CartPole-v0')
    env.seed(0)
    #  box = continuous
    print(env.observation_space)
    print(env.reset())  # [position, velocity, pole angle, pole velocity at tip]
    print('max values of state space: %s' % env.observation_space.high)
    print('min values of state space: %s' % env.observation_space.low)
    print()
    print(env.action_space)  # [0=push left, 1=push right]
    n_episodes, n_timesteps = 100, 50
    for episode in range(n_episodes):
        cumulative_reward = 0
        state = env.reset()
        for t in range(n_timesteps):
            env.render()
            random_action = env.action_space.sample()
            next_state, reward, absorbing_state, info = env.step(random_action)
            cumulative_reward += reward
            if absorbing_state:
                break
        if episode % 10 == 0:
            print('Episode %d, return %f' % (episode, cumulative_reward))
    env.close()


def atari_tennis(recording=False):
    env = gym.make('Tennis-v0')
    if recording:
        env = gym.wrappers.Monitor(env, 'recording', force=True)
        env.reset()
        for _ in range(5000):
            env.render()
            action = env.action_space.sample()
            next_state, reward, absorbing_state, info = env.step(action)
            if absorbing_state:
                break
        env.close()
        return
    n_episodes, n_timesteps = 100, 50
    for episode in range(n_episodes):
        cumulative_reward = 0
        env.reset()
        for t in range(n_timesteps):
            env.render()
            random_action = env.action_space.sample()
            next_state, reward, absorbing_state, info = env.step(random_action)
            cumulative_reward += reward
            if absorbing_state:
                break
        if episode % 10 == 0:
            print('Episode %d, return %f' % (episode, cumulative_reward))
    env.close()


def recording_test():
    env = gym.make('Tennis-v0')
    env = gym.wrappers.Monitor(env, 'recording', force=True)
    n_timesteps = 5000
    cumulative_reward = 0
    env.reset()
    for t in range(n_timesteps):
        env.render()
        random_action = env.action_space.sample()
        next_state, reward, absorbing_state, info = env.step(random_action)
        cumulative_reward += reward
        if absorbing_state:
            break
    env.close()


# atari_tennis(recording=True)
# recording_test()

# env = gym.make('Pong-v0')
# max_iterations = env._max_episode_steps










































































































































































































































































































































































