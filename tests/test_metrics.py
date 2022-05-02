from __future__ import annotations

from functools import partial
from typing import Optional

import numpy as np
from numpy.random import default_rng

from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from c2st.check import c2st as _compare

# All tests below assume default scoring="accuracy" instead of the new
# "balanced_accuracy", so bend that back here.
compare = partial(_compare, scoring="accuracy")

FIXEDSEED = 1309
NDIM = 10
NSAMPLES = 1024
RNG = default_rng(FIXEDSEED)


def _get_mlp_clf(ndim, random_state=None):
    return MLPClassifier(
        activation="relu",
        ##hidden_layer_sizes=(10 * ndim, 10 * ndim),
        hidden_layer_sizes=(ndim // 2,),
        max_iter=1000,
        solver="adam",
        random_state=random_state,
        early_stopping=True,
        learning_rate="adaptive",
    )


def old_compare(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int = FIXEDSEED,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    verbosity: int = 0,
) -> np.ndarray:

    return _compare(
        X=X,
        Y=Y,
        scoring=scoring,
        z_score=z_score,
        noise_scale=noise_scale,
        verbosity=verbosity,
        cv=KFold(n_splits=n_folds, random_state=seed, shuffle=True),
        clf=_get_mlp_clf(X.shape[1], random_state=seed),
    )


def old_c2st(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int = FIXEDSEED,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
) -> np.ndarray:
    if z_score:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * np.random.randn(X.shape)
        Y += noise_scale * np.random.randn(Y.shape)

    clf = _get_mlp_clf(ndim=X.shape[1], random_state=seed)
    data = np.concatenate((X, Y))
    target = np.concatenate((np.zeros((X.shape[0],)), np.ones((Y.shape[0],))))

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    scores = np.mean(scores)
    return np.atleast_1d(scores)


def test_old_and_partial():

    xnormal = partial(
        RNG.multivariate_normal, mean=np.zeros(NDIM), cov=np.eye(NDIM)
    )

    X = xnormal(size=(1024,))
    Y = xnormal(size=(1024,))

    obs_c2st = old_compare(X, Y)
    exp_c2st = old_c2st(X, Y)
    print("old_compare", obs_c2st)
    print("old_c2st", exp_c2st)

    assert np.allclose(obs_c2st, exp_c2st)


def test_same_distributions_alt():

    xnormal = partial(
        RNG.multivariate_normal, mean=np.zeros(NDIM), cov=np.eye(NDIM)
    )

    X = xnormal(size=(NSAMPLES,))
    Y = xnormal(size=(NSAMPLES,))

    obs_c2st = compare(X, Y)

    assert obs_c2st is not None
    assert (
        0.48 < obs_c2st < 0.52
    )  # only by chance we differentiate the 2 samples
    print(obs_c2st)


def test_diff_distributions_alt():

    xnormal = partial(
        RNG.multivariate_normal, mean=np.zeros(NDIM), cov=np.eye(NDIM)
    )
    ynormal = partial(
        RNG.multivariate_normal, mean=20.0 * np.ones(NDIM), cov=np.eye(NDIM)
    )

    X = xnormal(size=(NSAMPLES,))
    Y = ynormal(size=(NSAMPLES,))

    obs_c2st = compare(X, Y)

    assert obs_c2st is not None
    assert (
        0.98 < obs_c2st
    )  # distributions do not overlap, classifiers label with high accuracy
    print(obs_c2st)


def test_distributions_overlap_by_two_sigma_alt():

    xnormal = partial(
        RNG.multivariate_normal, mean=np.zeros(NDIM), cov=np.eye(NDIM)
    )
    ynormal = partial(
        RNG.multivariate_normal, mean=1.0 * np.ones(NDIM), cov=np.eye(NDIM)
    )

    X = xnormal(size=(NSAMPLES,))
    Y = ynormal(size=(NSAMPLES,))

    obs_c2st = compare(X, Y)

    assert obs_c2st is not None
    print(obs_c2st)
    assert (
        0.8 < obs_c2st
    )  # distributions do not overlap, classifiers label with high accuracy


def test_old_same_distributions_default():

    xnormal = partial(
        RNG.multivariate_normal, mean=np.zeros(NDIM), cov=np.eye(NDIM)
    )

    X = xnormal(size=(NSAMPLES,))
    Y = xnormal(size=(NSAMPLES,))

    obs_c2st = old_c2st(X, Y)

    assert obs_c2st is not None
    assert (
        0.49 < obs_c2st < 0.51
    )  # only by chance we differentiate the 2 samples


def test_old_same_distributions_default_flexible_alt():

    xnormal = partial(
        RNG.multivariate_normal, mean=np.zeros(NDIM), cov=np.eye(NDIM)
    )

    X = xnormal(size=(NSAMPLES,))
    Y = xnormal(size=(NSAMPLES,))

    seed = 42
    obs_c2st = old_c2st(X, Y, seed=seed)

    assert obs_c2st is not None
    assert (
        0.48 < obs_c2st[0] < 0.52
    )  # only by chance we differentiate the 2 samples

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    clf = _get_mlp_clf(ndim=X.shape[1], random_state=seed)
    obs2_c2st = compare(
        X,
        Y,
        clf=clf,
        cv=cv,
    )

    assert obs2_c2st is not None
    assert (
        0.48 < obs2_c2st < 0.52
    )  # only by chance we differentiate the 2 samples
    assert np.allclose(obs2_c2st, obs_c2st)


def test_old_diff_distributions_default():

    xnormal = partial(
        RNG.multivariate_normal, mean=np.zeros(NDIM), cov=np.eye(NDIM)
    )
    ynormal = partial(
        RNG.multivariate_normal, mean=20.0 * np.ones(NDIM), cov=np.eye(NDIM)
    )

    X = xnormal(size=(NSAMPLES,))
    Y = ynormal(size=(NSAMPLES,))

    obs_c2st = old_c2st(X, Y)

    assert obs_c2st is not None
    print(obs_c2st)
    assert (
        0.98 < obs_c2st
    )  # distributions do not overlap, classifiers label with high accuracy


def test_old_distributions_overlap_by_two_sigma_default():

    xnormal = partial(
        RNG.multivariate_normal, mean=np.zeros(NDIM), cov=np.eye(NDIM)
    )
    ynormal = partial(
        RNG.multivariate_normal, mean=1.0 * np.ones(NDIM), cov=np.eye(NDIM)
    )

    X = xnormal(size=(NSAMPLES,))
    Y = ynormal(size=(NSAMPLES,))

    obs_c2st = old_c2st(X, Y)

    assert obs_c2st is not None
    print(obs_c2st)
    assert (
        0.8 < obs_c2st
    )  # distributions do not overlap, classifiers label with high accuracy
