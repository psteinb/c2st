from typing import Sequence

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping


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

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.validation_fraction,
                shuffle=True,
                random_state=self.random_state,
            )
            eval_set = [(X_test, y_test)]
        else:
            eval_set = None
            X_train, y_train = X, y
        return super().fit(X_train, y_train, *args, eval_set=eval_set, **kwds)
