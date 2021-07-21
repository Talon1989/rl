import random
import gym
import numpy as np
from collections import deque
import tensorflow.keras as keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.optimizers import Adam


env = gym.make('MsPacman-v0')
n_states = (88, 80, 1)  # greyscale normalized image
n_actions = env.action_space.n


#  PREPROCESSING ##


color = np.array([210, 164, 74]).mean()


def preprocess_state(s):
    image = s[1:176:2, ::2]
    image = image.mean(axis=2)
    image[image == color] = 0
    image = ((image - 128) / 128) - 1
    image = np.expand_dims(image.reshape(88, 80, 1), axis=0)
    return image


#  DQN ##


class DQN:

    def __init__(self, state_size, action_size, gamma=0.9, epsilon=0.8):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_buffer = deque(maxlen=5_000)
        self.update_rate = 1_000
        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.update_target_network()

    def build_network(self):
        model = keras.Sequential([
            keras.layers.Conv2D(
                filters=32, kernel_size=(8, 8), strides=4,
                padding='same', input_shape=self.state_size, activation='relu'
            ),
            keras.layers.Conv2D(
                filters=64, kernel_size=(4, 4), strides=2, padding='same', activation='relu'
            ),
            keras.layers.Conv2D(
                filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'
            ),
            keras.layers.Flatten(),
            keras.layers.Dense(units=512, activation='relu'),
            keras.layers.Dense(units=self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam())
        return model

    def store_transition(self, s, a, r, s_, done):
        self.replay_buffer.append([s, a, r, s_, done])

    def epsilon_greedy(self, s):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            q_values = self.main_network.predict(s)
            return np.argmax(q_values[0])

    def train(self, batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size)
        # minibatch = np.random.choice(self.replay_buffer, batch_size)  # same as above with numpy
        for s, a, r, s_, done in minibatch:
            if not done:
                # 'np.amax' return the maximum of an array or maximum along an axis
                target_q = (r + self.gamma * np.amax(self.target_network.predict(s_)))
            else:
                target_q = r
            q_values = self.main_network.predict(s)
            q_values[0][a] = target_q
            self.main_network.fit(
                x=s, y=q_values, epochs=1, verbose=0
            )

    def update_target_network(self):
        self.target_network.set_weights(
            self.main_network.get_weights()
        )


#  training the dqn ##


n_episodes = 500
n_timesteps = 20_000
# batch_size = 8
batch_size = 32
n_screens = 4
dqn = DQN(state_size=n_states, action_size=n_actions)
done = False
time_step = 0
for i in range(n_episodes):
    return_ = 0
    s = preprocess_state(env.reset())
    # for t in range(n_timesteps):
    while True:
        env.render()
        time_step += 1
        if time_step % dqn.update_rate == 0:
            dqn.update_target_network()
        a = dqn.epsilon_greedy(s)
        s_, r, done, _ = env.step(a)
        s_ = preprocess_state(s_)
        dqn.store_transition(s, a, r, s_, done)
        s = s_
        return_ += r
        if done:
            print('Episode: %d, Return: %f' % (i, return_))
            break
        # if len(dqn.replay_buffer) > batch_size:
        #     dqn.train(batch_size=batch_size)
    if len(dqn.replay_buffer) > batch_size:
        print('training dqn')
        dqn.train(batch_size=batch_size)




















































































































































































































































