import gym
import numpy as np
import pandas as pd
from tensorflow import keras
from data.utils import plot_learning_curve


def onehot(y):
    from sklearn.preprocessing import LabelEncoder
    y = LabelEncoder().fit_transform(y)
    Y = np.zeros([np.unique(y).shape[0], y.shape[0]])
    for idx, val in enumerate(y):
        Y[int(val), idx] = 1
    return Y.T


class ReplayBuffer:

    def __init__(self, max_size, input_shape, n_actions, classification=False):
        self.mem_size = max_size
        self.classification = classification  # to use in case of onehot
        self.state_memory = np.zeros([self.mem_size, input_shape])
        self.new_state_memory = np.zeros([self.mem_size, input_shape])
        self.dtype = np.int8 if self.classification else np.float32
        self.action_memory = np.zeros([self.mem_size, n_actions], dtype=self.dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros([self.mem_size], dtype=np.float32)  # 0 for dones, 1 for not dones
        self.mem_cntr = 0

    def store_transition(self, s, a, r, s_, done):
        index = self.mem_cntr % self.mem_size  # modular to get back to index 0 at the end
        self.state_memory[index] = s
        self.action_memory[index] = a
        self.reward_memory[index] = r
        self.new_state_memory[index] = s_
        self.terminal_memory[index] = 1 - done
        if self.classification:  # only binary
            actions = np.zeros(self.action_memory.shape[1])
            actions[a] = 1
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = a
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):  # retrieves list of transition, not single sample
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        s = self.state_memory[batch]
        a = self.action_memory[batch]
        r = self.reward_memory[batch]
        s_ = self.new_state_memory[batch]
        terminals = self.terminal_memory[batch]
        return s, a, r, s_, terminals


def build_dqn(alpha, n_actions, input_dims, h1_dims, h2_dims):
    model = keras.models.Sequential([
        keras.layers.Dense(units=h1_dims, input_shape=(input_dims, ), activation='relu'),
        keras.layers.Dense(units=h2_dims, activation='relu'),
        keras.layers.Dense(units=n_actions)
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr=alpha), loss='mse')
    return model


class Agent:

    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=999/1000, epsilon_min=1/100,
                 mem_size=1_000_000, fname='z_train_dqn_00_model.h5'):
        self.alpha = alpha
        self.gamma = gamma
        self.n_actions = n_actions
        # self.action_space = np.arange(self.n_actions)  ##################
        self.action_space = [i for i in range(n_actions)]
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.model_file = 'data/' + fname
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, classification=True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256)

    def remember(self, s, a, r, s_, done):
        self.memory.store_transition(s, a, r, s_, done)

    def choose_action(self, s):
        s = s[np.newaxis, :]
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            return np.argmax(self.q_eval.predict(s))

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        s, a, r, s_, done = self.memory.sample_buffer(self.batch_size)
        action_values = np.array([i for i in range(self.n_actions)], dtype=np.int8)
        action_indices = np.dot(a, action_values)
        q_eval = self.q_eval.predict(s)
        q_next = self.q_eval.predict(s_)
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, action_indices] = r + self.gamma * np.max(q_next, axis=1) * done
        self.q_eval.fit(s, q_target, verbose=False)
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = keras.models.load_model(self.model_file)


# flowers = pd.read_csv('data/iris.csv').iloc[:, -1].values
# flowers = onehot(flowers)


env = gym.make('LunarLander-v2')
n_games = 1_000
agent = Agent(
    alpha=0.0005, gamma=0.99, n_actions=env.action_space.n,
    epsilon=1., batch_size=64, input_dims=8,
)
# agent.load_model()
scores = []
eps_history = []
training = False


if training:
    for i in range(n_games):
        score = 0
        s = env.reset()
        while True:
            env.render()
            a = agent.choose_action(s)
            s_, r, done, _ = env.step(a)
            score += r
            agent.remember(s, a, r, s_, done)
            s = s_
            agent.learn()
            if done:
                break
        eps_history.append(agent.epsilon)
        scores.append(score)
        avg_score = np.float(np.mean(scores[max(0, i-100): i+1]))
        print(
            'episode %d, score %.2f, avg score %.2f' % (i, score, avg_score)
        )
        if i % 100 == 0 and i > 0:
            agent.save_model()


else:
    agent.load_model()
    agent.epsilon = agent.epsilon_min
    for i in range(n_games):
        score = 0
        s = env.reset()
        while True:
            env.render()
            a = agent.choose_action(s)
            s_, r, done, _ = env.step(a)
            score += r
            s = s_
            if done:
                print('episode %d, score %.2f' % (i, score))
                break


# x = [i+1 for i in range(n_games)]
# plot_learning_curve(x, scores, eps_history, filename='data/lunarlander.png')






















































































































































































































































































