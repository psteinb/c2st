"""
sklearn.neural_network.MLPRegressor-like API using skorch as NN backend.
"""

from collections import OrderedDict

import numpy as np

import torch as T
import skorch
from skorch.classifier import NeuralNetClassifier, NeuralNetBinaryClassifier


class _SkorchClassifierModule(T.nn.Module):
    """Torch NN model for usage as skorch classifier module."""

    is_binary = None

    def __init__(
        self,
        ndim_in=None,
        hidden_layer_sizes=(100,),
        batch_norm=False,
        activation=T.nn.ReLU,
    ):
        """
        is_binary : bool
            True -> for NeuralNetBinaryClassifier, using BCEWithLogitsLoss
            False -> for NeuralNetClassifier, using NLLLoss
        """
        super().__init__()

        assert ndim_in is not None
        self.ndim_in = ndim_in
        assert self.is_binary is not None

        ndim_out = 1 if self.is_binary else 2
        layers = []
        layers.append(
            ("linear_in", T.nn.Linear(ndim_in, hidden_layer_sizes[0]))
        )
        if batch_norm:
            layers.append(
                ("batch_norm", T.nn.BatchNorm1d(hidden_layer_sizes[0]))
            )
        layers.append(("act_0", activation()))
        for ii, size in enumerate(hidden_layer_sizes[1:]):
            ii_layer = ii + 1
            layers.append(
                (
                    f"linear_{ii_layer}",
                    T.nn.Linear(hidden_layer_sizes[ii], size),
                )
            )
            layers.append((f"act_{ii_layer}", activation()))

        layers.append(
            (
                f"linear_out",
                T.nn.Linear(hidden_layer_sizes[-1], ndim_out),
            )
        )

        if not self.is_binary:
            layers.append((f"softmax", T.nn.Softmax(dim=-1)))

        self.model = T.nn.Sequential(OrderedDict(layers))

    def forward(self, X):
        assert self.ndim_in == X.shape[1]
        return self.model(X)


# Since the used torch loss funcs are very picky re. the dtypes they accept, we
# need to define them here.
class SkorchClassifierModule(_SkorchClassifierModule):
    # NLLLoss
    dtype_data = np.float32
    dtype_target = np.int64
    is_binary = False


class SkorchBinaryClassifierModule(_SkorchClassifierModule):
    # BCEWithLogitsLoss
    dtype_data = np.float32
    dtype_target = np.float32
    is_binary = True


# Defaults matching most of sklearn.neural_network.MLPRegressor's defaults,
# except:
#
# * batch_size: sklearn uses min(200, n_samples), we'd need to define a custom
#   fit() to support setting this at runtime since we don't know n_samples
#   until fit() gets data passed in
# * max_iter
# * weight decay: sklearn uses alpha > 0, we use Adam(weight_decay=0), so no
#   weight_decay. Reason: sklearn says "The L2 regularization term is divided
#   by the sample size when added to the loss.". Looking at the code, "sample
#   size" = batch_size but again it would be a bit of work to support the same
#   behavior using skorch's API. We'd need to set weight_decay=alpha/batch_size
#   at runtime once we know batch_size, not impossible but not our main concern
#   ATM either. So instead of choosing a different non-zero default, we disable
#   it. Also we actually recommend using AdamW over Adam if you want
#   weight_decay, so use
#
#     skorch_*_classifier(optimizer=AdamW,
#                         optimizer__weight_decay=1e-4/batch_size)
#   or
#
#     skorch_*_classifier(optimizer=Adam,
#                         optimizer__weight_decay=1e-4/batch_size)
#
#   to match sklearn's behavior.
#
# Equivalent param names:
#
# sklearn                   skorch
# -------                   ------
# learning_rate_init        lr
# n_iter_no_change          patience
# alpha                     Adam(weight_decay=alpha/batch_size)
# validation_fraction=0.1   ValidSplit(0.1)
# max_iter                  max_epochs      # for adam and sgd
#
default_kwds = dict(
    max_epochs=10000,
    lr=1e-3,
    batch_size=32,
    optimizer=T.optim.Adam,
    train_split=skorch.dataset.ValidSplit(0.1),
    verbose=False,
    callbacks=[
        skorch.callbacks.EarlyStopping(
            monitor="valid_acc",
            patience=10,
            threshold=1e-4,
            threshold_mode="abs",
            lower_is_better=False,
        )
    ],
    iterator_train__shuffle=True,
    module__hidden_layer_sizes=(100,),
    module__batch_norm=False,
)
# Adam(weight_decay=0) should be torch default anyway, but just to be safe;
# note that AdamW's default weight_decay=1e-2
default_kwds["optimizer__weight_decay"] = 0


def skorch_classifier(*args, **kwds):
    _kwds = default_kwds.copy()
    _kwds.update(kwds)
    return NeuralNetClassifier(SkorchClassifierModule, *args, **_kwds)


def skorch_binary_classifier(*args, **kwds):
    _kwds = default_kwds.copy()
    _kwds.update(kwds)
    return NeuralNetBinaryClassifier(
        SkorchBinaryClassifierModule, *args, **_kwds
    )
