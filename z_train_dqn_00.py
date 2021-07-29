import numpy as np
import pandas as pd
from tensorflow import keras


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
        self.action_memory = np.zeros([self.mem_size, n_actions])
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


# flowers = pd.read_csv('data/iris.csv').iloc[:, -1].values
# flowers = onehot(flowers)























































































































































































































































































