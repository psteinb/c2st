import numpy as np
from sklearn.model_selection import train_test_split

import pytest

from c2st.utils import train_test_split_idx


@pytest.mark.parametrize(
    "shuffle,stratify", [(True, None), (True, "y"), (False, None)]
)
def test_train_test_split_idx(shuffle, stratify):
    rng = np.random.default_rng()
    X = rng.random(size=(1000, 10))
    y = np.concatenate((np.zeros(500), np.ones(500)))
    seed = 123

    kwds = dict(
        test_size=0.2,
        shuffle=shuffle,
        random_state=seed,
        stratify=y if stratify == "y" else None,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, **kwds)
    idxs_train, idxs_test = train_test_split_idx(X, y, **kwds)

    assert (X_train == X[idxs_train]).all()
    assert (y_train == y[idxs_train]).all()
    assert (X_test == X[idxs_test]).all()
    assert (y_test == y[idxs_test]).all()
