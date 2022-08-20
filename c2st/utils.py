import numpy as np
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


def train_test_split_idx(
    X, y, test_size=0.2, shuffle=True, stratify=None, random_state=None
):
    """Implements a subset of sklearn.model_selection.train_test_split() but
    only returns train and test array indices instead of data copies to save
    memory.
    """
    assert X.ndim == 2
    assert y.ndim == 1
    assert test_size > 0, "test_size must be > 0"
    n_samples = X.shape[0]
    assert n_samples == len(y), "X and y must have equal length"
    n_test = int(test_size * n_samples)
    assert n_test > 0, f"{n_test=}, increase test_size"
    n_train = n_samples - n_test

    if not shuffle:
        idxs_train = np.arange(n_train)
        idxs_test = np.arange(n_train, n_train + n_test)
    else:
        if stratify is not None:
            CVClass = StratifiedShuffleSplit
        else:
            CVClass = ShuffleSplit

        cv = CVClass(
            test_size=n_test, train_size=n_train, random_state=random_state
        )

        idxs_train, idxs_test = next(cv.split(X=X, y=stratify))
    return idxs_train, idxs_test
