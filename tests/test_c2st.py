from typing import Sequence

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold

import pytest

from c2st.check import c2st, alpha2score, score2pvalue


rng = np.random.default_rng(seed=123)


def randn(*size):
    return rng.normal(loc=0, scale=1, size=size)


def test_api():
    X = randn(10, 3)
    Y = randn(5, 3)

    scores_mean = c2st(X, Y)
    assert isinstance(scores_mean, float)

    scores_mean = c2st(X, Y, clf=KNeighborsClassifier(5))
    scores_mean = c2st(X, Y, cv=StratifiedKFold())
    scores_mean = c2st(X, Y, noise_scale=0.1)

    scores_mean, scores = c2st(X, Y, return_scores=True, cv=KFold(7))
    assert len(scores) == 7
    assert isinstance(scores, np.ndarray)

    scores_mean, _, clfs = c2st(
        X, Y, return_scores=True, return_clfs=True, cv=KFold(7)
    )
    assert isinstance(scores_mean, float)
    assert isinstance(clfs, Sequence)
    assert len(clfs) == 7
    assert isinstance(clfs[0], RandomForestClassifier)

    scores_mean, clfs = c2st(
        X, Y, return_clfs=True, cv=KFold(7), clf=KNeighborsClassifier(5)
    )
    assert isinstance(scores_mean, float)
    assert isinstance(clfs, Sequence)
    assert len(clfs) == 7
    assert isinstance(clfs[0], KNeighborsClassifier)


def test_api_deprecated_pass():
    X = randn(10, 3)
    Y = randn(5, 3)

    ##with pytest.warns(DeprecationWarning):
    with pytest.deprecated_call():
        c2st(X, Y, cross_val_score_kwds=dict())


@pytest.mark.xfail
def test_api_deprecated_fail():
    X = randn(10, 3)
    Y = randn(5, 3)

    c2st(
        X,
        Y,
        cross_val_kwds=dict(verbose=2),
        cross_val_score_kwds=dict(verbose=2),
    )


def test_alpha():
    score = 0.5 + 0.1 * rng.random()
    test_size = 1e3
    score_roundtrip = alpha2score(score2pvalue(score, test_size), test_size)
    np.testing.assert_allclose(score, score_roundtrip, rtol=0, atol=1e-12)
