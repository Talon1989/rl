import numpy as np
import pandas as pd


iris = pd.read_csv('data/iris.csv')
X_ = iris.iloc[:, :-1].values
a = np.cov(X_.T)


def covariance_matrix(M):
    for f in range(M.shape[1]):
        feature_mean = np.mean(M[:, f])
        M[:, f] = M[:, f] - feature_mean
    return np.dot(M.T, M) / (M.shape[0] - 1)


b = covariance_matrix(X_)



































































































































































































































































































































