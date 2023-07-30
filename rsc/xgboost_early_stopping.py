"""
Notes on how to use XGBClassifier with EarlyStopping and how to use our
EarlyStoppingXGBClassifier wrapper.

loss: XGBClassifier(objective=...)
==================================

default: logloss for classification

https://xgboost.readthedocs.io/en/stable/parameter.html
https://scikit-learn.org/stable/modules/model_evaluation.html#log-loss
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss

tl;dr that's the (binary) cross-entropy loss.

Probably also the same as "objective=binary:logistic" since using the latter or
nothing (default is logloss) produces the exact same numbers.

early stopping
==============

With a callback EarlyStopping(metric_name=...)

    XGBClassifier(
        callbacks=[EarlyStopping(rounds=5, metric_name="error")],
        eval_metric=["error", "logloss"],
        ...,
        )
we have maximal control over early stopping.


When we use

    XGBClassifier(
        early_stopping_rounds=5,
        eval_metric=["error", "logloss"],
        ...
        )

instead then eval_metric[-1] (if it is a list of multiple metrics) is used for
early stopping ("logloss" in the example above). If not set, then the default
metric for early stopping is objective: rmse for regression, logloss for
classification, mean average precision for ranking.

Use large n_estimators (= # iterations = # trees generated) and rely on early
stopping.

fit()
=====
With

  eval_set=[(X_train, y_train), (X_test, y_test)],

we have

eval_set[ii] -> validation_{ii} key in evals_result_ dict, so validation_0 is
metrics on train set, validation_1 on test set. Which metrics are calculated is
defined by XGBClassifier(eval_metric=...).

For early stopping, the metric used for that (e.g.
EarlyStopping(metric_name="logloss")) is calculated from last entry in
eval_set, so here the test set.

error and accuracy
==================
error in xgboost is defined as: Binary classification error rate. It is
calculated as #(wrong cases)/#(all cases).

With binary classification accuracy

    A = (TN + TP) / (TN + TP + FN + FP)
      =: T / (T + F)

we have

    E = F / (T + F)

such that

    E + A = 1
    A = 1 - E

Therefore monitoring falling xgboost error in early stopping is the same as
monitoring rising test/validation accuracy in skorch or sklearn MLP
classifiers.
"""


import numpy as np
from matplotlib import pyplot as plt

from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from c2st.check import c2st
from c2st.classifiers.xgboost import EarlyStoppingXGBClassifier


def get_es_callback():
    # We cannot re-use EarlyStopping instances between XGBClassifier instances,
    # so we need to create them afresh each time.
    return EarlyStopping(
        rounds=5,
        metric_name="error",
        # Look for minimal test error or maximal test accuracy = 1-error.
        maximize=False,
        # Return best of the last 5 models, else we always get the 5th (for
        # rounds=5).
        save_best=True,
        # This is an absolute threshold. That's OK if metric_name="error"
        # which is bounded to [0,1] and interpretable.
        min_delta=1e-3,
    )


if __name__ == "__main__":
    seed = 123
    rng = np.random.default_rng(seed=seed)

    print("rng")
    N = int(1e3)
    D = 100
    P = rng.normal(loc=0, scale=1, size=(N, D))
    Q = rng.normal(loc=0, scale=1, size=(N, D))

    # ------------------------------------------------------------------------
    # Train single XGBClassifier model, test EarlyStopping, use standard
    # XGBClassifier and pass in eval_set to fit()
    # ------------------------------------------------------------------------

    clf = XGBClassifier(
        n_estimators=500,
        eval_metric=["error", "logloss"],
        tree_method="hist",
        # early_stopping_rounds: Switch on early stopping with default behavior
        # (return last model, not best, etc) when not using EarlyStopping()
        # callback to fine-tune.
        ##early_stopping_rounds=5,
        callbacks=[get_es_callback()],
        random_state=seed,
    )

    print("concatenate")
    X = np.concatenate((P, Q), axis=0)
    y = np.concatenate((np.zeros(N), np.ones(N)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True, random_state=seed
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.1, shuffle=True, random_state=seed
    )

    X_data = dict(train=X_train, valid=X_valid, test=X_test)
    y_data = dict(train=y_train, valid=y_valid, test=y_test)

    use = ["train", "test", "valid"]
    ##use = ["train", "test"]

    print("fit")
    clf.fit(
        X_data["train"],
        y_data["train"],
        eval_set=[(X_data[name], y_data[name]) for name in use],
        verbose=True,
    )

    for name in use:
        print(
            f"{name} acc  :",
            accuracy_score(y_data[name], clf.predict(X_data[name])),
        )
        print(f"{name} score:", clf.score(X_data[name], y_data[name]))

    metrics = dict()
    for idx, name in enumerate(use):
        for kind in ["error", "logloss"]:
            metrics[f"{name}_{kind}"] = np.array(
                clf.evals_result_[f"validation_{idx}"][kind]
            )

    for name in use:
        metrics[f"{name}_acc"] = 1 - metrics[f"{name}_error"]

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
    x = range(len(list(metrics.values())[0]))
    for ax_idx, kind in enumerate(["acc", "error", "logloss"]):
        ax = axs.flat[ax_idx]
        for name in use:
            ax.plot(x, metrics[f"{name}_{kind}"], label=f"{name} {kind}")

    for ax in axs.flat:
        ax.legend()
    plt.show()

    # ------------------------------------------------------------------------
    # Use EarlyStoppingXGBClassifier w/ EarlyStopping in c2st
    # ------------------------------------------------------------------------

    clf = EarlyStoppingXGBClassifier(
        n_estimators=500,
        eval_metric=["error", "logloss"],
        tree_method="hist",
        callbacks=[get_es_callback()],
        validation_fraction=0.1,
        random_state=seed,
    )

    scores, scores_mean = c2st(
        P,
        Q,
        clf=clf,
        return_scores=True,
        z_score=False,
    )

    print(scores_mean)
    print(scores)
