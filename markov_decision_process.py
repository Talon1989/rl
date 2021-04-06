import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


m_ = 3
m2_ = m_ ** 2
q = np.zeros(m2_)
q[m2_ // 2] = 1


def get_P(m, p_up, p_down, p_left, p_right):
    m2 = m**2
    P = np.zeros([m2, m2])
    ix_map = {i + 1: (i // m, i % m) for i in range(m2)}
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


P_ = get_P(3, 0.2, 0.3, 0.25, 0.25)
Pn = np.linalg.matrix_power(P_, 1)
result_ = np.matmul(q, Pn)



















































































































































































































































































































































































































































