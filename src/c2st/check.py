from typing import Union, Tuple
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

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
    nan_drop: bool = False,
    dtype_data=None,
    dtype_target=None,
    cross_val_score_kwds: dict = dict(),
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Return accuracy of classifier trained to distinguish samples from
    supposedly two distributions <X> and <Y>. For details on the method, see
    [1,2]. If the returned accuracy is 0.5, <X> and <Y> are considered to be
    from the same generating PDF, i.e. they can not be differentiated. If the
    returned accuracy is around 1., <X> and <Y> are considered to be from two
    different generating PDFs.

    Trains classifiers with N-fold cross-validation [3]. By default, a
    `RandomForestClassifier` from scikit-learn is used. This can be changed by
    passing another classifier instance, e.g.

    ::

        c2st(..., clf=KNeighborsClassifier(5))

    Also other CV methods can be specified that way.

    ::

        c2st(..., cv=StratifiedKFold(5))

    Args:
        X: Samples from one distribution, shape (n_samples_X, n_features)
        Y: Samples from another distribution, shape (n_samples_Y, n_features)
        scoring: a classifier scoring metric, anything that
            sklearn.model_selection.cross_val_score(scoring=...) accepts
        z_score: Z-scoring using X
        noise_scale: If passed, will add Gaussian noise with std noise_scale to
            samples of X and of Y
        verbose: control the verbosity of
            sklearn.model_selection.cross_val_score
        clf: classifier class instance with sklearn-compatible API, e.g.
            sklearn.ensemble.RandomForestClassifier
        cv: cross-validation class instance with sklearn-compatible API, e.g.
            sklearn.model_selection.KFold
        return_scores: Return 1d array of CV scores in addition to their mean
        nan_drop: Filter NaNs from CV scores and at least return the mean of
            the values left in scores
        dtype_data: numpy dtype for data=concatenate((X,Y)), default is X's dtype
        dtype_target: numpy dtype for target=concatenate((zeros(..),
            ones(..))), default is numpy's float default (np.float64)
        cross_val_score_kwds: Additional kwds passed to sklearn's
            cross_val_score()

    Returns:
        mean_scores: Mean of the accuracy scores over the test sets from
            cross-validation.
        scores: 1d array of CV scores. Only if return_scores is True.

    References:
        [1]: http://arxiv.org/abs/1610.06545
        [2]: https://www.osti.gov/biblio/826696/
        [3]: https://scikit-learn.org/stable/modules/cross_validation.html
    """
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

    scores = cross_val_score(
        clf,
        data,
        target,
        cv=cv,
        scoring=scoring,
        verbose=verbose,
        **cross_val_score_kwds,
    )

    if nan_drop:
        isnan = np.isnan(scores)
        if isnan.any():
            scores = scores[~isnan]
        if len(scores) == 0:
            warnings.warn("Only NaN scores, return NaN")
            if return_scores:
                return np.nan, np.array([np.nan] * len(isnan))
            else:
                return np.nan
    mean_scores = scores.mean()
    if return_scores:
        return mean_scores, scores
    else:
        return mean_scores


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

    [1] Lopez-Paz et al., ICLR 2017, https://arxiv.org/abs/1610.06545
    """
    return 0.5 + norm.ppf(1 - alpha) / np.sqrt(4 * test_size)


def score2pvalue(score, test_size):
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