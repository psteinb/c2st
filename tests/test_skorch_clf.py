import importlib

import pytest
import numpy as np

##from c2st.classifiers.skorch import skorch_classifier, skorch_binary_classifier


have_torch = importlib.util.find_spec("torch") is not None
have_skorch = importlib.util.find_spec("skorch") is not None


@pytest.mark.skipif(not have_torch, reason="torch not found")
@pytest.mark.skipif(not have_skorch, reason="skorch not found")
@pytest.mark.parametrize(
    "build_clf_name", ["skorch_classifier", "skorch_binary_classifier"]
)
def test_skorch_clf(build_clf_name):
    c2st_classifiers_skorch = importlib.import_module(
        "c2st.classifiers.skorch"
    )
    build_clf = getattr(c2st_classifiers_skorch, build_clf_name)
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
