import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt


def matrix_power(A: np.array, n):
    if n <= 0:
        return np.identity(A.shape[0])
    current_A = np.copy(A)
    for i in range(n-1):
        current_A = np.dot(current_A, A)
    return current_A


m_ = 3
m2_ = m_ ** 2
q = np.zeros(m2_)
q[m2_ // 2] = 1


def get_P(m, p_up, p_down, p_left, p_right):
    m2 = m**2
    P = np.zeros([m2, m2])
    ix_map = {i + 1: (i // m, i % m) for i in range(m2)}
    # print(ix_map)
    for i in range(m2):
        for j in range(m2):
            r1, c1 = ix_map[i + 1]
            r2, c2 = ix_map[j + 1]
            rdiff = r1 - r2
            cdiff = c1 - c2
            if rdiff == 0:
                if cdiff == 1:
                    P[i, j] = p_left
                elif cdiff == -1:
                    P[i, j] = p_right
                elif cdiff == 0:
                    if r1 == 0:
                        P[i, j] += p_down
                    elif r1 == m - 1:
                        P[i, j] += p_up
                    if c1 == 0:
                        P[i, j] += p_left
                    elif c1 == m - 1:
                        P[i, j] += p_right
            elif rdiff == 1:
                if cdiff == 0:
                    P[i, j] = p_down
            elif rdiff == -1:
                if cdiff == 0:
                    P[i, j] = p_up
    return P


P_ = get_P(m=3, p_up=0.2, p_down=0.3, p_left=0.25, p_right=0.25)
#  ALERT ! MATRIX IS BEING ROTATED SUCH THAT COLS HAVE PROBABILITIES
P_ = P_.T
Pn = matrix_power(P_, 1)  # Pn = np.linalg.matrix_power(P_, 1)
result_ = np.dot(Pn, q)
# result_ = np.dot(q, Pn)  # result_ = np.matmul(q, Pn)

# visual operations
# result_matrix = result_.reshape([3, 3])
# last_row = np.copy(result_matrix[-1, :])
# result_matrix[-1, :] = result_matrix[0, :]
# result_matrix[0, :] = last_row


# a sample path in an ergodic Markov chain
def path_in_ergodic_markov_chain():
    np.random.seed(0)
    s = 0
    n = 10 ** 3
    visited = [s]
    for t in range(n):
        s = np.random.choice(m2_, p=P_[:, s])  # np.random.choice(3, p=[0.1, 0.5, 0.4])
        visited.append(s)
    # plt.plot(visited[-20:], c='blue', linewidth=1); plt.show(); plt.clf();
    # return stats.itemfreq(visited)  # deprecated
    return np.unique(visited, return_counts=True)


# num_visited = np.vstack([path_in_ergodic_markov_chain()]).T


#  MARKOV REWARD PROCESS


















































































































































































































































































































































































































































