import random
import gym
import numpy as np
from tensorflow import keras
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
tf.disable_v2_behavior()


env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
n_actions = env.action_space.n
gamma = 0.95

def discount_and_normalize_rewards(ep_rewards):
    discounted_rewards = np.zeros_like(ep_rewards)
    reward_to_go = 0.
    for i in reversed(range(len(ep_rewards))):
        reward_to_go = reward_to_go * gamma + ep_rewards[i]
        discounted_rewards[i] = reward_to_go
    # normalization
    discounted_rewards = discounted_rewards - np.mean(discounted_rewards)
    discounted_rewards = discounted_rewards / np.std(discounted_rewards)
    return discounted_rewards


#  BUILDING POLICY NETWORK

state_ph = tf.placeholder(tf.float32, [None, state_size], name='state_ph')
action_ph = tf.placeholder(tf.int32, [None, n_actions], name='action_ph')
discounted_rewards_ph = tf.placeholder(tf.float32, [None, ], name='discounted_rewards')

layer_1 = tf.layers.dense(state_ph, units=32, activation=tf.nn.relu)
layer_2 = tf.layers.dense(layer_1, units=n_actions)

proba_distr = tf.nn.softmax(layer_2)

neg_log_policy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_2, labels=action_ph)
loss = tf.reduce_mean(neg_log_policy * discounted_rewards_ph)

train = tf.train.AdamOptimizer(0.01).minimize(loss)


#  TRAINING THE NETWORK

n_iter = 1_000
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(n_iter):
        episode_states, episode_actions, episode_rewards = [], [], []
        Return = 0
        s = env.reset()
        while True:
            s = s.reshape([1, 4])
            pi = session.run(proba_distr, feed_dict={state_ph: s})
            a = np.random.choice(range(pi.shape[1]), p=pi.ravel())
            s_, r, done, _ = env.step(a)
            env.render()
            Return += r
            action = np.zeros(n_actions)
            action[a] = 1
            episode_states.append(s)
            episode_actions.append(action)
            episode_rewards.append(r)
            if done:
                break
            s = s_
        discounted_rewards = discount_and_normalize_rewards(episode_rewards)
        feed_dict = {
            state_ph: np.vstack(np.array(episode_states)),
            action_ph: np.vstack(np.array(episode_actions)),
            discounted_rewards_ph: discounted_rewards
        }
        loss_, _ = session.run([loss, train], feed_dict=feed_dict)
        if i % 10 == 0:
            print('Iteration: %d, Return: %d' % (i, Return))









































































































































































































































































































































