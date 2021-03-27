import numpy as np
from sklearn.preprocessing import LabelEncoder


colors = [
    'blue', 'yellow', 'green', 'green', 'blue', 'yellow', 'blue', 'yellow', 'blue', 'green'
]
colors = np.array(colors)
values = np.array([
    'a', 'a', 'b', 'b', 'a', 'a', 'a', 'a', 'a', 'b'
])
data = np.column_stack([colors, values])


def onehot(z):  # works with any type of data
    uniques = np.unique(z)
    numerical_data = np.arange(uniques.shape[0])
    Z = np.zeros([uniques.shape[0], z.shape[0]])
    uniques = list(uniques)
    for idx, val in enumerate(z):
        Z[numerical_data[uniques.index(val)], idx] = 1
    return Z.T


z_ = onehot(data[:, 0])
numeric_values = LabelEncoder().fit_transform(data[:, -1])
data_ = np.column_stack([z_, numeric_values])









