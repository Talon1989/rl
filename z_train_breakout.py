import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from baselines_bruteforce.atari_wrappers import make_atari, wrap_deepmind


# env = gym.make('BreakoutNoFrameskip-v4')
# n_actions = env.action_space.n
# color = np.array([210, 164, 74]).mean()
#
#
# def preprocess_state(s):
#     image = s[1:176:2, ::2]
#     image = image.mean(axis=2)
#     image[image == color] = 0
#     image = ((image - 128) / 128) - 1
#     image = np.expand_dims(image.reshape(88, 80, 1), axis=0)
#     return image
#
#
# def random_play(n_games=1):
#     env.reset()
#     counter = 0
#     while True:
#         action = np.random.randint(0, n_actions)
#         _, _, done, _ = env.step(action)
#         counter += done
#         env.render()
#         if counter == n_games:
#             break
#         if done:
#             print('Ending game %d' % counter)
#             env.reset()


# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (  # Rate at which to reduce chance of random action being taken
        epsilon_max - epsilon_min
)
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000


# Use the Baseline Atari environment because of Deepmind helper functions
env = make_atari("BreakoutNoFrameskip-v4")
# Warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=True, scale=True)
n_actions = env.action_space.n


def q_model():
    return keras.models.Sequential([
        keras.layers.Conv2D(
            filters=32, kernel_size=8, strides=4, activation='relu',
            padding='valid', input_shape=(84, 84, 4)
        ),
        keras.layers.Conv2D(
            filters=64, kernel_size=4, strides=2, activation='relu'
        ),
        keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=1, activation='relu'
        ),
        keras.layers.Flatten(),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dense(units=n_actions, activation='linear')
    ])


model_main = q_model()
model_target = q_model()


optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.)
loss_function = keras.losses.Huber()

action_history, state_history, state_next_history = [], [], []
rewards_history, done_history, episode_reward_history = [], [], []
running_reward, episode_count, frame_count = 0, 0, 0
#  number of frames to take random action and observe output
epsilon_random_frames = 50_000
#  number of frames for exploration
epsilon_greedy_frames = 1_000_000.
#  maximum replay length
max_memory_length = 100_000
update_after_action = 4
update_target_network_rate = 10_000


def e_greedy(n_frames, s):
    if frame_count < epsilon_random_frames or np.random.uniform(0, 1) < epsilon:
        a = np.random.randint(0, n_actions)
    else:
        state_tensor = tf.expand_dims(tf.convert_to_tensor(s), 0)
        action_probas = model_main(state_tensor, training=False)
        a = tf.argmax(action_probas[0]).numpy()
    return a


def update_main_network():

    #  get indices of samples for replay buffers
    indices = np.random.choice(range(len(done_history)), size=batch_size)

    #  using list comprehension to sample from replay buffer
    state_sample = np.array([state_history[i] for i in indices])
    state_next_sample = np.array([state_next_history[i] for i in indices])
    rewards_sample = [rewards_history[i] for i in indices]
    action_sample = [action_history[i] for i in indices]
    done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

    #  build the updated q-values for the sampled future states
    future_rewards = model_target.predict(state_next_sample)
    updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)

    #  if final frame then set the last value to -1
    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

    #  create a mask to calculate loss only updated q-values
    masks = tf.one_hot(action_sample, n_actions)

    with tf.GradientTape() as tape:
        q_values = model_main(state_sample)
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = loss_function(updated_q_values, q_action)

    #  backpropagation
    gradients = tape.gradient(loss, model_main.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_main.trainable_variables))


def update_target_network():
    model_target.set_weights(model_main.get_weights())
    print(
        'running reward: %.2f at episode %d, frame count %d'
        % (running_reward, episode_count, frame_count)
    )


while True:

    s = np.array(env.reset())
    episode_reward = 0

    for t in range(1, max_steps_per_episode):

        env.render()
        if frame_count % 1000 == 0:
            print('Episode %d, frame %d' % (episode_count, frame_count))

        frame_count += 1
        a = e_greedy(frame_count, s)

        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon_random_frames = max(epsilon, epsilon_min)

        s_, r, done, _ = env.step(a)
        s_ = np.array(s_)

        episode_reward += r

        action_history.append(a)
        state_history.append(s)
        state_next_history.append(s_)
        done_history.append(done)
        rewards_history.append(r)
        s = s_

        if frame_count % update_after_action == 0 and len(done_history) > batch_size:
            update_main_network()

        if frame_count % update_target_network_rate == 0:
            update_target_network()

        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    if running_reward > 40:
        print('Solved at episode %d' % episode_count)
        break
























































































































































































































































































































































