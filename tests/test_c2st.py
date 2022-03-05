import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, KFold

from c2st.check import c2st, c2st_


rng = np.random.default_rng(seed=123)


def randn(*size):
    return rng.normal(loc=0, scale=1, size=size)


def test_api():
    X = randn(10, 3)
    Y = randn(5, 3)

    ms = c2st(X, Y, clf=KNeighborsClassifier(5))
    ms = c2st(X, Y, cv=StratifiedKFold())

    ms, s = c2st(X, Y, return_scores=True, cv=KFold(7))
    assert len(s) == 7


def test_api_underscore():
    X = randn(10, 3)
    Y = randn(5, 3)

    s = c2st_(X, Y, cv=KFold(7))
    assert len(s) == 7
