import tensorflow as tf
print(tf.__version__)
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import gym
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


env = gym.make('CartPole-v0')
state_shape = env.observation_space.shape[0]
num_actions = env.action_space.n
gamma = 0.95

def discount_and_normalize_rewards(episode_rewards):
    discounted_rewards = np.zeros_like(episode_rewards)
    reward_to_go = 0.0
    for i in reversed(range(len(episode_rewards))):
        reward_to_go = reward_to_go * gamma + episode_rewards[i]
        discounted_rewards[i] = reward_to_go
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    return discounted_rewards


state_ph = tf.placeholder(tf.float32, [None, state_shape], name="state_ph")
action_ph = tf.placeholder(tf.int32, [None, num_actions], name="action_ph")
discounted_rewards_ph = tf.placeholder(tf.float32, [None,], name="discounted_rewards")
layer1 = tf.layers.dense(state_ph, units=32, activation=tf.nn.relu)
layer2 = tf.layers.dense(layer1, units=num_actions)
prob_dist = tf.nn.softmax(layer2)
neg_log_policy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = layer2, labels = action_ph)
loss = tf.reduce_mean(neg_log_policy * discounted_rewards_ph)
train = tf.train.AdamOptimizer(0.01).minimize(loss)


num_iterations = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_iterations):
        episode_states, episode_actions, episode_rewards = [],[],[]
        done = False
        state = env.reset()
        Return = 0
        while not done:
            state = state.reshape([1,4])
            pi = sess.run(prob_dist, feed_dict={state_ph: state})
            a = np.random.choice(range(pi.shape[1]), p=pi.ravel())
            next_state, reward, done, info = env.step(a)
            env.render()
            Return += reward
            action = np.zeros(num_actions)
            action[a] = 1
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            state=next_state
        discounted_rewards= discount_and_normalize_rewards(episode_rewards)
        feed_dict = {state_ph: np.vstack(np.array(episode_states)),
                     action_ph: np.vstack(np.array(episode_actions)),
                     discounted_rewards_ph: discounted_rewards
                     }

        #train the network
        loss_, _ = sess.run([loss, train], feed_dict=feed_dict)

        #print the return for every 10 iteration
        if i%10==0:
            print("Iteration:{}, Return: {}".format(i,Return))



