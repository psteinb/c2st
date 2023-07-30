from typing import Sequence

from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from c2st.utils import train_test_split_idx


def has_es_callback(callbacks: Sequence):
    assert isinstance(callbacks, Sequence), f"{callbacks=} is not a sequence"
    for obj in callbacks:
        if isinstance(obj, EarlyStopping):
            return True
    return False


class EarlyStoppingXGBClassifier(XGBClassifier):
    """XGBClassifier wrapper with automatic eval_set creation in ``fit()``
    based on `validation_fraction`.
    """

    def __init__(
        self,
        *,
        validation_fraction: float = 0.1,
        validation_split_shuffle=True,
        **kwds,
    ):
        assert not hasattr(
            self, "validation_fraction"
        ), "validation_fraction attribute already present"

        if validation_fraction is not None:
            if not (
                "early_stopping_rounds" in kwds
                or has_es_callback(kwds.get("callbacks", []))
            ):
                raise ValueError(
                    "validation_fraction specified but none of "
                    "early_stopping_rounds or EarlyStopping callback given"
                )
        self.validation_fraction = validation_fraction
        self.validation_split_shuffle = validation_split_shuffle

        return super().__init__(**kwds)

    def fit(self, X, y, *args, **kwds):
        if self.validation_fraction is not None:
            if "eval_set" in kwds:
                raise ValueError(
                    "validation_fraction implies automatic generation of "
                    "eval_set, but eval_set was specified "
                )

            idxs_train, idxs_test = train_test_split_idx(
                X,
                y,
                test_size=self.validation_fraction,
                shuffle=self.validation_split_shuffle,
                random_state=self.random_state,
            )
            return super().fit(
                X[idxs_train, :],
                y[idxs_train],
                *args,
                eval_set=[(X[idxs_test, :], y[idxs_test])],
                **kwds,
            )
        else:
            return super().fit(X, y, *args, **kwds)

def get_es_callback():
    return EarlyStopping(
        rounds=5,
        metric_name="error",
        maximize=False,
        save_best=True,
        min_delta=1e-3,
    )


default_kwds = dict(
    n_estimators=500,
    eval_metric=["error", "logloss"],
    tree_method="hist",
    random_state=123,
    validation_fraction=0.1,
    validation_split_shuffle=True,
)


def get_clf(*args, **kwds):
    _kwds = default_kwds.copy()
    _kwds["callbacks"] = [get_es_callback()]
    _kwds.update(kwds)
    return EarlyStoppingXGBClassifier(*args, **_kwds)
