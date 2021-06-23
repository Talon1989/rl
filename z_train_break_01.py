import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# test = np.array([[1, 2, 3], [2, 1, 3]])
# print(np.cov(test))


iris = pd.read_csv('data/iris.csv')


def covariance_matrix(M: np.array):
    def normalization(A):
        means = np.sum(A, axis=0) / A.shape[0]
        print(means)
        return A - means
    norm_M = normalization(M)
    return np.dot(
        norm_M.T, norm_M
    ) / (norm_M.shape[0] - 1)


X_ = iris.iloc[:, :-1].values
# cov_1 = np.cov(X_.T)
# cov_2 = covariance_matrix(X_)


class NaiveBayesClassifier:
    # TODO refine

    def __init__(self, X: np.array, y: np.array):
        assert X.shape[0] == y.shape[0]
        n_classes = len(np.unique(y))
        summation = np.zeros([n_classes, X.shape[1]])
        for i in range(X.shape[0]):
            summation[y[i]] += X[i]
        means = []
        for c in range(n_classes):
            means.append(summation[c] / len(y[y == c]))
        means = np.array(means)
        variances = np.zeros([n_classes, X.shape[1]])
        for i in range(X.shape[0]):
            variances[y[i]] += (X[i] - means[y[i]])**2
        self.n_classes = n_classes
        self.means = means
        self.variances = variances
        self.proportion_classes = np.array(
            [len(y[y == c]) for c in range(n_classes)]
        ) / y.shape[0]

    def predict(self, X):

        def func(x, c):
            posteriors = np.sum(
                np.log(
                    1/(np.sqrt(2*np.pi * self.variances[c])) *
                    np.exp(-((x - self.means[c]) ** 2)/(2 * self.variances[c]))
                ),
                axis=1
            )
            return posteriors + np.log(self.proportion_classes[c])

        preds = []
        for c in range(self.n_classes):
            preds.append(func(X, c))
        preds = np.array(preds)
        # print(preds)
        return np.argmax(
            preds, axis=0
        )


y_ = LabelEncoder().fit_transform(iris.iloc[:, -1])
# X_ = np.append(X_, np.array([5.9, 3.1, 5.2, 1.7]), axis=0)
X_ = np.vstack([X_, np.array([5.9, 3.1, 5.2, 1.7])])
X_ = RobustScaler().fit_transform(X_)
y_ = np.append(y_, 2)

X_train, X_test, y_train, y_test = train_test_split(
    X_, y_, stratify=y_, train_size=0.5
)

classifier = NaiveBayesClassifier(X_train, y_train)

predictions = classifier.predict(X_test)
print(
    np.sum(predictions == y_test) / y_test.shape[0]
)
print(classifier.predict(X_test[-10:-1]))
print(y_test[-10:-1])




































































































































































































































































































































































































