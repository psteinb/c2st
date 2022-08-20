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


class C2STXGBClassifier(XGBClassifier):
    def __init__(self, *, validation_fraction: float = 0.1, **kwds):
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
                shuffle=True,
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
