from typing import Optional
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score


def c2st(
    X: np.ndarray,
    Y: np.ndarray,
    seed=None,
    n_folds=None,
    scoring: str = "balanced_accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    verbosity: int = 0,
    clf_class=None,
    clf_kwargs=None,
    clf=RandomForestClassifier(random_state=1),
    cv=KFold(n_splits=5, shuffle=True, random_state=1),
    return_scores: bool = False,
    nan_drop=False,
) -> np.ndarray:
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
        z_score: Z-scoring using X
        noise_scale: If passed, will add Gaussian noise with std noise_scale to
            samples of X and of Y
        verbosity: control the verbosity of
            sklearn.model_selection.cross_val_score
        clf: a scikit-learn classifier class instance
        cv: cross-validation class instance with sklearn API, e.g.
            sklearn.model_selection.KFold
        return_scores: Return 1d array of CV scores in addition to their mean
        nan_drop: Filter NaNs from CV scores and at least return the mean of
            the values left in scores.

    Returns:
        mean_scores: Mean of the accuracy scores over the test sets from
            cross-validation.
        scores: 1d array of CV scores. Only if return_scores is True.

    References:
        [1]: http://arxiv.org/abs/1610.06545
        [2]: https://www.osti.gov/biblio/826696/
        [3]: https://scikit-learn.org/stable/modules/cross_validation.html
    """

    # Support previous API
    kwds = dict(
        seed=seed, clf_class=clf_class, clf_kwargs=clf_kwargs, n_folds=n_folds
    )

    def _get(key, default):
        val = kwds.get(key)
        if val is not None:
            warnings.warn(f"{key} deprecated", DeprecationWarning)
            return val
        else:
            return default

    # If any of the kwds are used, switch to previous API.
    if list(kwds.values()) != [None] * len(kwds):
        clf_class = _get("clf_class", RandomForestClassifier)
        clf_kwargs = _get("clf_kwargs", dict())
        seed = _get("seed", 1)
        n_folds = _get("n_folds", 5)
        clf = clf_class(random_state=seed, **clf_kwargs)
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    if z_score:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * np.random.randn(*X.shape)
        Y += noise_scale * np.random.randn(*Y.shape)

    # prepare data
    data = np.concatenate((X, Y))
    # labels
    target = np.concatenate((np.zeros((X.shape[0],)), np.ones((Y.shape[0],))))

    scores = cross_val_score(
        clf, data, target, cv=cv, scoring=scoring, verbose=verbosity
    )

    if nan_drop:
        isnan = np.isnan(scores)
        if isnan.any():
            scores = scores[~isnan]
        if len(scores) == 0:
            warnings.warn(f"Only NaN scores, return NaN")
            if return_scores:
                return np.nan, np.array([np.nan] * len(isnan))
            else:
                return np.nan
    mean_scores = scores.mean()
    if return_scores:
        return mean_scores, scores
    else:
        return mean_scores


# Support previous API
def c2st_(*args, **kwds):
    """Same as c2st(..., return_scores=True)[1], so return only a 1d array of
    CV scores. Args and kwds are the same as c2st().
    """
    return c2st(*args, **kwds, return_scores=True)[1]
