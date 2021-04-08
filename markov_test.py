import numpy as np


#  probability distribution matrix
M = np.array([
    [0.3, 0.5, 0.2],
    [0.5, 0.1, 0.1],
    [0.2, 0.4, 0.7]
])

#  initial probability vector
p_0 = np.array([0.8, 0.1, 0.1])


def matrix_power(A, n):
    if n <= 0:
        return np.identity(A.shape[0])
    current_A = np.copy(A)
    for i in range(n-1):
        current_A = np.dot(current_A, A)
    return current_A


def matrix_diagonalization(A):
    lambdas, V = np.linalg.eig(A)
    L = np.zeros([A.shape[0], A.shape[1]])
    for i in range(A.shape[0]):
        L[i, i] = lambdas[i]
    return np.dot(
        np.dot(V, L),
        np.linalg.inv(V)
    )
def matrix_diagonalization_powers(A: np.array, n: int):
    assert n >= 0
    lambdas, V = np.linalg.eig(A)
    L = np.zeros([A.shape[0], A.shape[1]])
    for i in range(A.shape[0]):
        L[i, i] = lambdas[i]
    if n >= 1:
        return np.dot(
            np.dot(V, L ** n),
            np.linalg.inv(V)
        )
    else:
        # return np.dot(
        #     np.dot(V, np.identity(A.shape[0])),
        #     np.linalg.inv(V)
        # )
        return np.identity(A.shape[0])


def matrix_diagonalization_and_initial_vector(A, v_0):
    lambdas, V = np.linalg.eig(A)
    L = np.zeros([A.shape[0], A.shape[1]])
    for i in range(A.shape[0]):
        L[i, i] = lambdas[i]
    c = np.dot(np.linalg.inv(V), v_0)
    return np.dot(
        np.dot(V, L),
        np.linalg.inv(V)
    )
def matrix_diagonalization_and_initial_vector_powers(A, v_0, n):
    lambdas, V = np.linalg.eig(A)
    L = np.zeros([A.shape[0], A.shape[1]])
    for i in range(A.shape[0]):
        L[i, i] = lambdas[i]
    c = np.dot(np.linalg.inv(V), v_0)
    if n >= 1:
        return np.dot(
            np.dot(V, L ** n),
            c
        )
    else:
        return np.dot(
            V, c
        )
















































































































































































































































































