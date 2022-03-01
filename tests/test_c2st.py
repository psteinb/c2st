import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, KFold

from c2st.check import c2st


rng = np.random.default_rng(seed=123)


def randn(*size):
    return rng.normal(loc=0, scale=1, size=size)


def test_api():
    X = randn(1000, 3)
    Y = randn(100, 3)

    ms = c2st(X, Y)
    assert ms.ndim == 1
    assert len(ms) == 1

    ms = c2st(X, Y, clf=KNeighborsClassifier(5))
    ms = c2st(X, Y, cv=StratifiedKFold())

    ms, s = c2st(X, Y, return_scores=True, cv=KFold(7))
    assert len(s) == 7
