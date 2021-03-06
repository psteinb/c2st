import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, KFold

from c2st.check import c2st


rng = np.random.default_rng(seed=123)


def randn(*size):
    return rng.normal(loc=0, scale=1, size=size)


def test_api():
    X = randn(10, 3)
    Y = randn(5, 3)

    ms = c2st(X, Y, clf=KNeighborsClassifier(5))
    ms = c2st(X, Y, cv=StratifiedKFold())
    ms = c2st(X, Y, noise_scale=0.1)

    ms, s = c2st(X, Y, return_scores=True, cv=KFold(7))
    assert len(s) == 7
