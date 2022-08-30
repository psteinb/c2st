import importlib

import pytest
import numpy as np

from c2st.check import c2st
##from c2st.classifiers.skorch import skorch_classifier, skorch_binary_classifier


have_torch = importlib.util.find_spec("torch") is not None
have_skorch = importlib.util.find_spec("skorch") is not None


@pytest.mark.skipif(not have_torch, reason="torch not found")
@pytest.mark.skipif(not have_skorch, reason="skorch not found")
@pytest.mark.parametrize(
    "build_clf_name", ["get_clf", "get_binary_clf"]
)
def test_skorch_clf(build_clf_name):
    c2st_classifiers_skorch = importlib.import_module(
        "c2st.classifiers.skorch"
    )
    build_clf = getattr(c2st_classifiers_skorch, build_clf_name)
    ndim = 3

    def get_clf():
        return build_clf(
            module__ndim_in=ndim,
            module__hidden_layer_sizes=(20, 30),
            module__batch_norm=True,
            lr=1e-3,
            batch_size=10,
            verbose=True,
        )

    npoints = 100

    P = np.random.rand(npoints, ndim)
    Q = np.random.rand(npoints, ndim)
    X = np.concatenate((P, Q))
    y = np.concatenate((np.zeros(npoints), np.ones(npoints)))

    clf = get_clf()

    # Test API w/ auto type cast for np.ndarray input
    clf.fit(X, y)
    clf.predict(X)
    clf.predict_proba(X)
    clf.score(X, y)
    clf.partial_fit(X, y)

    # Test usage in c2st().
    scores, scores_mean = c2st(
        P,
        Q,
        clf=get_clf(),
        return_scores=True,
        z_score=False,
    )
