"""
sklearn.neural_network.MLPClassifier-like API using skorch as NN backend,
defining sensible defaults and shortcut functions for creating a classifier.


Passing args and keywords
=========================

With skorch, we define a Module which is the NN model representing our
classifier. This gets wrapped by skorch's NeuralNet{Binary}Classifier.
The pre-defined modules we implement here are called

SkorchClassifierModule          -> for NeuralNetClassifier
SkorchBinaryClassifierModule    -> for NeuralNetBinaryClassifier

Many settings like optimizer, lr, batch_size can be passes to
NeuralNet{Binary}Classifier directly. Other kwds for the wrapped module must be
passed as module__some_kwd=foo, which will do
Skorch{Binary}ClassifierModule(..., some_kwd=foo). This is important in
particular for custom kwds like hidden_layer_sizes in our case here.

    clf = NeuralNetClassifier(
        SkorchClassifierModule,
        optimizer=AdamW,
        lr=1e-3,
        module__hidden_layer_sizes=(100,),
        module__batch_norm=False,
        ...
        )
    clf.fit(X_train, y_train)
    clf.predict(X_test)

We have shortcut functions for the above with sensible defaults that match
sklearn's MLPClassifier as much as possible (see next section).

    clf = skorch_classifier()
    clf.fit(X_train, y_train)
    clf.predict(X_test)


Differences to sklearn
======================

default_kwds (see in code below): Defaults matching most of
sklearn.neural_network.MLPClassifier's defaults, except

* batch_size: sklearn uses min(200, n_samples), we'd need to define a custom
  fit() to support setting this at runtime since we don't know n_samples
  until fit() gets data passed in
* max_iter is called max_epochs and our default is higher
* weight decay: sklearn uses alpha > 0, we use Adam(weight_decay=0), so no
  weight_decay. Reason: sklearn says "The L2 regularization term is divided
  by the sample size when added to the loss.". Looking at the code, "sample
  size" = batch_size but again it would be a bit of work to support the same
  behavior using skorch's API. We'd need to set weight_decay=alpha/batch_size
  at runtime once we know batch_size, not impossible but not our main concern
  ATM either. So instead of choosing a different non-zero default, we disable
  it. Also we actually recommend using AdamW over Adam if you want
  weight_decay, so use

    skorch_*_classifier(optimizer=AdamW,
                        optimizer__weight_decay=1e-4/batch_size)
  or

    skorch_*_classifier(optimizer=Adam,
                        optimizer__weight_decay=1e-4/batch_size)

  to match sklearn's behavior.

Equivalent param names:

sklearn                   skorch
-------                   ------
learning_rate_init        lr
n_iter_no_change          patience
alpha                     Adam(weight_decay=alpha/batch_size)
validation_fraction=0.1   ValidSplit(0.1)
max_iter                  max_epochs      # for adam and sgd


Loss functions used by NeuralNet{Binary}Classifier
==================================================

tl;dr in both cases, we effectively use cross-entropy loss.

See https://github.com/skorch-dev/skorch/blob/master/skorch/toy.py:
make_classifier() vs. make_binary_classifier(). In the general multi-class case
we need to add softmax as last layer to output probabilities. From the skorch
help:

NeuralNetClassifier uses NLLLoss
  criterion : torch criterion (class, default=torch.nn.NLLLoss)
      Negative log likelihood loss. Note that the module should
      return probabilities, the log is applied during get_loss.

So the "module" (the NN model we define and wrap with NeuralNetClassifier())
must output probabilities, created by a last softmax layer.

"the log is applied in get_loss": From
https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss:

  The input given through a forward call is expected to contain
  log-probabilities of each class.
  ...
  Obtaining log-probabilities in a neural network is easily achieved by
  adding a LogSoftmax layer in the last layer of your network. You may
  use CrossEntropyLoss instead, if you prefer not to add an extra
  layer.

So skorch replaces

    NLLLoss(LogSoftmax(...))

by something like

    NLLLoss(get_loss(Softmax(...))

Well ok. This means what we in effect use here is CrossEntropyLoss.

NeuralNetBinaryClassifier uses BCEWithLogitsLoss
  criterion : torch criterion (class, default=torch.nn.BCEWithLogitsLoss)
      Binary cross entropy loss with logits. Note that the module should
      return the logit of probabilities with shape (batch_size, ).

From https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html:
    (BCE = binary cross entropy) This loss combines a Sigmoid layer and the
    BCELoss in one single class. This version is more numerically stable than
    using a plain Sigmoid followed by a BCELoss as, by combining the operations
    into one layer, we take advantage of the log-sum-exp trick for numerical
    stability.

"logit of probabilities": In the binary case, the softmax layer can be
rewritten to apply a sigmoid (=logistic function) to a single output instead of
two (see Murphy 2022, book1, ch. 2.5.2). Since applying sigmoid(x) =
1/(1+exp(-x)) to the last layer "produces probabilities" (well, clamps between
[0,1]) and the logit function logit(x)=ln x/(1-x) is the inverse of sigmoid(x),
the output of the last layer before applying sigmoid() is often called
"logits". Therefore, we don't apply softmax to the last layer of the Module
that we wrap with NeuralNetBinaryClassifier, meaning that "logit of
probabilities" is just the last layer outputs themselves.
"""

from collections import OrderedDict

import numpy as np

import torch as T
import skorch
from skorch.classifier import NeuralNetClassifier, NeuralNetBinaryClassifier


class _SkorchClassifierModule(T.nn.Module):
    """Torch NN model for usage as skorch classifier module."""

    # Bool, to be overwritten by derived classes.
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

        assert ndim_in is not None, "ndim_in is None"
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
        assert (
            self.ndim_in == X.shape[1]
        ), f"{self.ndim_in=} doesn't match {X.shape[1]=}"
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


def astype(x, dtype):
    if isinstance(x, np.ndarray):
        return x if x.dtype == dtype else x.astype(dtype)
    else:
        return x


class DtypeHandlerMixin:
    def predict(self, X, **kwds):
        return super().predict(
            astype(X, self.module.dtype_data),
            **kwds,
        )

    def predict_proba(self, X, **kwds):
        return super().predict_proba(
            astype(X, self.module.dtype_data),
            **kwds,
        )

    # fit_loop -> partial_fit -> fit
    def fit_loop(self, X, y, **kwds):
        return super().fit_loop(
            astype(X, self.module.dtype_data),
            astype(y, self.module.dtype_target),
            **kwds,
        )


class DtypeHandlerNeuralNetClassifier(DtypeHandlerMixin, NeuralNetClassifier):
    pass


class DtypeHandlerNeuralNetBinaryClassifier(
    DtypeHandlerMixin, NeuralNetBinaryClassifier
):
    pass


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
# note that AdamW's default is weight_decay=1e-2
default_kwds["optimizer__weight_decay"] = 0


def skorch_classifier(*args, **kwds):
    _kwds = default_kwds.copy()
    _kwds.update(kwds)
    return DtypeHandlerNeuralNetClassifier(
        SkorchClassifierModule, *args, **_kwds
    )


def skorch_binary_classifier(*args, **kwds):
    _kwds = default_kwds.copy()
    _kwds.update(kwds)
    return DtypeHandlerNeuralNetBinaryClassifier(
        SkorchBinaryClassifierModule, *args, **_kwds
    )
