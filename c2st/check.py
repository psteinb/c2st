# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

from typing import Optional
from functools import wraps

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

""" a numpy only implementation """

# this is a numpy only impl
def c2st_(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int = 1,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    verbosity: int = 0,
    clf_class=RandomForestClassifier,
    clf_kwargs={},
) -> np.ndarray:
    """
    Return accuracy of classifier trained to distinguish samples from supposedly
    two distributions <X> and <Y>. For details on the method, see [1,2].
    If the returned accuracy is 0.5, <X> and <Y> are considered to be from the
    same generating PDF, i.e. they can not be differentiated.
    If the returned accuracy is around 1., <X> and <Y> are considered to be from
    two different generating PDFs.

    Trains classifiers with N-fold cross-validation [3]. By default, a `RandomForestClassifier`
    by scikit-learn is used. This can be adopted using <clf_class> and
    <clf_kwargs> as in:

    ``` py
    clf = clf_class(random_state=seed, **clf_kwargs)
    #...
    scores = cross_val_score(
        clf, data, target, cv=shuffle, scoring=scoring, verbose=verbosity
    )
    ```

    Args:
        X: Samples from one distribution.
        Y: Samples from another distribution.
        seed: Seed for sklearn
        n_folds: Number of folds
        z_score: Z-scoring using X
        noise_scale: If passed, will add Gaussian noise with std noise_scale to samples of X and of Y
        verbosity: control the verbosity of sklearn.model_selection.cross_val_score
        clf_class: a scikit-learn classifier class
        clf_kwargs: key-value arguments dictuinary to the class specified by clf_class, e.g. sklearn.ensemble.RandomForestClassifier

    Return:
        np.ndarray offering the accuracy scores over the test sets from cross-validation

    Example:
    ``` py
    > c2st(X,Y)
    [0.51904464]
    #X and Y likely come from the same PDF or ensemble
    ```
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
        X += noise_scale * np.randn(X.shape)
        Y += noise_scale * np.randn(Y.shape)

    # X = X.cpu().numpy()
    # Y = Y.cpu().numpy()

    clf = clf_class(random_state=seed, **clf_kwargs)

    # prepare data
    data = np.concatenate((X, Y))
    # labels
    target = np.concatenate((np.zeros((X.shape[0],)), np.ones((Y.shape[0],))))

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(
        clf, data, target, cv=shuffle, scoring=scoring, verbose=verbosity
    )

    return scores


@wraps(c2st_)
def c2st(*args, **kwds) -> np.ndarray:
    """
    Return accuracy of classifier trained to distinguish samples from supposedly
    two distributions <X> and <Y>. For details on the method, see [1,2].
    If the returned accuracy is 0.5, <X> and <Y> are considered to be from the
    same generating PDF, i.e. they can not be differentiated.
    If the returned accuracy is around 1., <X> and <Y> are considered to be from
    two different generating PDFs.

    Trains classifiers with N-fold cross-validation [3]. By default, a `RandomForestClassifier`
    by scikit-learn is used. This can be adopted using <clf_class> and
    <clf_kwargs> as in:

    ``` py
    clf = clf_class(random_state=seed, **clf_kwargs)
    #...
    scores = cross_val_score(
        clf, data, target, cv=shuffle, scoring=scoring, verbose=verbosity
    )
    ```

    Args:
        X: Samples from one distribution.
        Y: Samples from another distribution.
        seed: Seed for sklearn
        n_folds: Number of folds
        z_score: Z-scoring using X
        noise_scale: If passed, will add Gaussian noise with std noise_scale to samples of X and of Y
        verbosity: control the verbosity of sklearn.model_selection.cross_val_score
        clf_class: a scikit-learn classifier class
        clf_kwargs: key-value arguments dictuinary to the class specified by clf_class, e.g. sklearn.ensemble.RandomForestClassifier

    Return:
        np.ndarray offering the accuracy scores over the test sets from cross-validation

    Example:
    ``` py
    > c2st(X,Y)
    [0.51904464]
    #X and Y likely come from the same PDF or ensemble
    ```
    References:
        [1]: http://arxiv.org/abs/1610.06545
        [2]: https://www.osti.gov/biblio/826696/
        [3]: https://scikit-learn.org/stable/modules/cross_validation.html
    """

    scores = np.asarray(np.mean(c2st_(*args, **kwds)), dtype=np.float32)
    return np.atleast_1d(scores)
