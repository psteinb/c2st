from typing import Union, Tuple, Sequence
import warnings
import copy

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_validate
from sklearn.base import ClassifierMixin

from scipy.stats import norm


def c2st(
    X: np.ndarray,
    Y: np.ndarray,
    scoring: str = "balanced_accuracy",
    z_score: bool = True,
    noise_scale: float = None,
    verbose: int = 0,
    clf=RandomForestClassifier(random_state=1),
    cv=KFold(n_splits=5, shuffle=True, random_state=1),
    return_scores: bool = False,
    return_clfs: bool = False,
    nan_drop: bool = False,
    dtype_data=None,
    dtype_target=None,
    cross_val_kwds: dict = dict(),
    cross_val_score_kwds: dict = None,
) -> Union[
    float,
    Tuple[float, np.ndarray],
    Tuple[float, Sequence[ClassifierMixin]],
    Tuple[float, np.ndarray, Sequence[ClassifierMixin]],
]:
    """
    Run the c2st method on samples in X and Y.

    Train k classifiers of type `clf` with k-fold cross-validation and return
    the mean of k scores as the c2st test statistic.

    By default, a `RandomForestClassifier` from scikit-learn is used. This can
    be changed by passing another classifier instance, e.g. `c2st(...,
    clf=KNeighborsClassifier(5))` Also other cross-validation methods can be
    specified that way: `c2st(..., cv=StratifiedKFold(5))`.

    Args:
        X: Samples from one distribution, shape (n_samples_X, n_features)
        Y: Samples from another distribution, shape (n_samples_Y, n_features)
        scoring: a classifier scoring metric, anything that
            sklearn.model_selection.cross_val_score(scoring=...) accepts
        z_score: Z-scoring using X: apply X's scaling also to Y, same as

            >>> from sklearn.preprocessing import StandardScaler
            >>> x_scaler=StandardScaler().fit(X)
            >>> X_scaled=x_scaler.transform(X)
            >>> Y_scaled=x_scaler.transform(Y)

        noise_scale: If passed, will add Gaussian noise with std noise_scale to
            samples of X and of Y
        verbose: control the verbosity of sklearn.model_selection.cross_validate
        clf: classifier class instance with sklearn-compatible API, e.g.
            sklearn.ensemble.RandomForestClassifier
        cv: cross-validation class instance with sklearn-compatible API, e.g.
            sklearn.model_selection.KFold
        return_scores: Return 1d array of CV scores in addition to their mean
        return_clfs: Return sequence of trained classifiers for each fold.
            This is equal to
            ``cross_validate(..., return_estimator=True)["estimator"]``
        nan_drop: Filter NaNs from CV scores and at least return the mean of
            the values left in scores
        dtype_data: numpy dtype for data=concatenate((X,Y)), default is X's dtype
        dtype_target: numpy dtype for target=concatenate((zeros(..),
            ones(..))), default is numpy's float default (np.float64)
        cross_val_kwds: Additional kwds passed to sklearn's
            cross_validate()

    Returns:
        mean_scores: Mean of the accuracy scores over the test sets from
            cross-validation.
        scores: 1d array of CV scores. Only if return_scores is True.
        clfs: Sequence of trained `clf` instances for each CV fold.
            Only if return_clfs is True.
    """
    if cross_val_score_kwds is not None:
        assert len(cross_val_kwds) == 0, (
            f"Cannot use cross_val_kwds and cross_val_score_kwds. "
            f"Got {cross_val_score_kwds=} {cross_val_kwds=}. "
            f"Use only cross_val_kwds."
        )

        # cross_validate() and cross_val_score() have basically the same input
        # signature, so this should be safe.
        warnings.warn(
            "cross_val_score_kwds is deprecated, use "
            "cross_val_kwds. Will pass cross_val_score_kwds to "
            "cross_validate().",
            DeprecationWarning,
        )

        # Copy to avoid altering passed in dict in return_clfs=True case.
        _cross_val_kwds = copy.deepcopy(cross_val_score_kwds)
    else:
        _cross_val_kwds = copy.deepcopy(cross_val_kwds)

    if z_score:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * np.random.randn(*X.shape)
        Y += noise_scale * np.random.randn(*Y.shape)

    assert X.dtype == Y.dtype, f"{X.dtype=} not equal to {Y.dtype=}"

    data = np.concatenate((X, Y), axis=0)
    if dtype_data is not None:
        data = data.astype(dtype_data)

    target = np.concatenate((np.zeros((X.shape[0],)), np.ones((Y.shape[0],))))
    if dtype_target is not None:
        target = target.astype(dtype_target)

    if return_clfs and (not _cross_val_kwds.get("return_estimator", False)):
        _cross_val_kwds["return_estimator"] = True

    cv_data = cross_validate(
        clf,
        data,
        target,
        cv=cv,
        scoring=scoring,
        verbose=verbose,
        **_cross_val_kwds,
    )
    scores = cv_data["test_score"]

    if nan_drop:
        isnan = np.isnan(scores)
        if isnan.any():
            scores = scores[~isnan]
        if len(scores) == 0:
            warnings.warn("Only NaN scores, return NaN")
            scores = np.array([np.nan] * len(isnan))

    mean_scores = scores.mean()
    out = (mean_scores,)
    if return_scores:
        out += (scores,)
    if return_clfs:
        out += (cv_data["estimator"],)
    return out[0] if len(out) == 1 else out


def alpha2score(alpha: float, test_size: Union[int, float]):
    """Convert significance level alpha (e.g. alpha=0.05) to maximal c2st score
    t = 0.5 + epsilon below which we accept the null hypothesis P=Q where X~P
    and Y~Q.

    This is z_alpha from [1], appendix B, using the null distribution of the
    test statistic t ~ N(1/2, 1/(4*test_size)), where t = c2st score [0.5...1].

    Args:
        alpha: significance level
        test_size: size of test set, for the default cv=KFold(5) we do here
            this would be (X.shape[0] + Y.shape[0])/5

    Returns:
        max_score: Score below which we accept the null hypothesis P=Q.

    References:
        [1] Lopez-Paz et al., ICLR 2017, https://arxiv.org/abs/1610.06545
    """
    return 0.5 + norm.ppf(1 - alpha) / np.sqrt(4 * test_size)


def score2pvalue(score: float, test_size: Union[int, float]):
    """Inverse of alpha2score().

    Convert c2st score (test statistic t) to p-value. If this is bigger than a
    chosen significance level (e.g. alpha=0.05) then we accept the null
    hypothesis P=Q, else we reject it.

    The p-value is the probablilty, given P=Q where t=0.5, that we observe
    score t=0.5+epsilon or bigger. This probablilty gets smaller with
    increasing t since it is more unlikely to get scores far away from 0.5 when
    P=Q is true.

    Example:

    >>> from c2st import check
    >>> alpha=0.05
    # Assume X.shape[0] + Y.shape[0] = 1e4
    >>> test_size=1e4/5
    >>> for alpha in [0.05, 0.01]:
    ...     for score in [0.51, 0.52, 0.53]:
    ...         p = check.score2pvalue(score, test_size)
    ...         print(f"{score=}, {p=:.4f}",
                    f"accept (p>{alpha}): P = Q" if p>=alpha else
                    f"reject (p<{alpha}): P != Q")
    score=0.51, p=0.1855 accept (p>0.05): P = Q
    score=0.52, p=0.0368 reject (p<0.05): P != Q
    score=0.53, p=0.0036 reject (p<0.05): P != Q
    score=0.51, p=0.1855 accept (p>0.01): P = Q
    score=0.52, p=0.0368 accept (p>0.01): P = Q
    score=0.53, p=0.0036 reject (p<0.01): P != Q
    """
    return 1 - norm.cdf(score, loc=0.5, scale=0.5 * np.sqrt(1 / test_size))
