import importlib

import pytest
import numpy as np

have_xgboost = importlib.util.find_spec("xgboost") is not None


@pytest.mark.skipif(not have_xgboost, reason="xgboost not found")
def test_xgboost_clf():
    from xgboost.callback import EarlyStopping

    from c2st.check import c2st
    from c2st.classifiers.xgboost import C2STXGBClassifier

    def get_es_callback():
        return EarlyStopping(
            rounds=5,
            metric_name="error",
            maximize=False,
            save_best=True,
            min_delta=1e-3,
        )

    def get_clf():
        return C2STXGBClassifier(
            n_estimators=500,
            eval_metric=["error", "logloss"],
            tree_method="hist",
            callbacks=[get_es_callback()],
            validation_fraction=0.1,
            random_state=seed,
        )

    ndim = 3
    npoints = 100
    seed = 123

    P = np.random.rand(npoints, ndim)
    Q = np.random.rand(npoints, ndim)
    X = np.concatenate((P, Q))
    y = np.concatenate((np.zeros(npoints), np.ones(npoints)))

    clf = get_clf()

    # Test API
    clf.fit(X, y)
    clf.predict(X)
    clf.predict_proba(X)
    clf.score(X, y)

    # Check that EarlyStopping actually worked.
    assert hasattr(clf, "best_iteration")
    assert clf.best_iteration < 500
    assert set(clf.evals_result_.keys()) == set(["validation_0"])
    niter = len(clf.evals_result_["validation_0"]["error"])
    assert niter < 500
    assert len(clf.evals_result_["validation_0"]["logloss"]) == niter

    # Test usage in c2st().
    scores, scores_mean = c2st(
        P,
        Q,
        clf=get_clf(),
        return_scores=True,
        z_score=False,
    )
