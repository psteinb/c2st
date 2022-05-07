import pytest

import numpy as np
from c2st.classifiers.skorch import skorch_classifier, skorch_binary_classifier


@pytest.mark.parametrize(
    "build_clf", [skorch_classifier, skorch_binary_classifier]
)
def test_skorch_clf(build_clf):
    ndim = 3
    clf = build_clf(
        module__ndim_in=ndim,
        module__hidden_layer_sizes=(20, 30),
        module__batch_norm=True,
        lr=1e-3,
        batch_size=10,
        verbose=True,
    )
    X = np.random.rand(100, ndim).astype(clf.module.dtype_data)
    y = np.concatenate((np.ones(50), np.zeros(50))).astype(
        clf.module.dtype_target
    )
    clf.fit(X, y)
    clf.predict(X)
