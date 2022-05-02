# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     argv:
#     - /usr/bin/python3
#     - -m
#     - ipykernel_launcher
#     - -f
#     - '{connection_file}'
#     display_name: Python 3
#     env: null
#     interrupt_mode: signal
#     language: python
#     metadata: null
#     name: python3
# ---

# +
import sys
from pathlib import Path

reporoot = Path(".").absolute().parent

# -

sys.path.append(str(reporoot))


from c2st.check import c2st_


from __future__ import annotations
import numpy as np
from functools import partial
from numpy.random import default_rng
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn import __version__ as sklversion
import time
FIXEDSEED = 1328


# +
def nn_c2st_(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int = FIXEDSEED,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    verbosity: int = 0,
) -> np.ndarray:

    ndim = X.shape[1]
    clf_class = MLPClassifier
    clf_kwargs = {
        "activation": "relu",
        "hidden_layer_sizes": (10 * ndim, 10 * ndim),
        "max_iter": 1000,
        "solver": "adam",
    }

    return c2st_(
        X,
        Y,
        seed,
        n_folds,
        scoring,
        z_score,
        noise_scale,
        verbosity,
        clf_class,
        clf_kwargs,
    )

def early_c2st_(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int = FIXEDSEED,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    verbosity: int = 0,
) -> np.ndarray:

    ndim = X.shape[1]
    clf_class = MLPClassifier
    clf_kwargs = {
        "activation": "relu",
        "hidden_layer_sizes": (10 * ndim, 10 * ndim),
        "max_iter": 1000,
        "solver": "adam",
        "early_stopping": True,
        "n_iter_no_change": 50,
    }

    return c2st_(
        X,
        Y,
        seed,
        n_folds,
        scoring,
        z_score,
        noise_scale,
        verbosity,
        clf_class,
        clf_kwargs,
    )


# -

NDIM = 10
max_nsamples = 4048
sample_sizes = [ 2**it for it in range(7,10)]
sample_sizes.append(max_nsamples)
RNG = default_rng(FIXEDSEED)
print(sample_sizes)

# +
center_normal = partial(RNG.multivariate_normal, mean=np.zeros(NDIM), cov=np.eye(NDIM))
distributions = { 0. : center_normal}

for alpha in np.linspace(0,2,9):
    distributions[alpha] = partial(RNG.multivariate_normal, mean=np.zeros(NDIM) + alpha, cov=np.eye(NDIM))



# +
center_samples = center_normal(size=max_nsamples)
samples = {}
for k,v in distributions.items():
    samples[k] = v(size=max_nsamples)

assert samples[0.].shape == (max_nsamples, 10)

# +
rf_results = {}
rf_timings = {}
total = len(samples.values())*len(sample_sizes)
cnt = 0


for k,v in samples.items():
    for size in sample_sizes:

        try:
            start = time.time()
            scores = c2st_(X = center_samples[:size,...],
                          Y = v[:size,...],
                          n_folds=10)
            end = time.time()
        except Exception as ex:
            print(ex)
            continue

        if not k in rf_results.keys():
            rf_results[k] = []
            rf_timings[k] = []

        mean = np.mean(scores)
        std = np.std(scores)
        rf_results[k].append(list(scores))
        rf_timings[k].append(len(scores)*[end-start])
        cnt += 1

        print(f"{cnt}/{total}: {k}[{size},...] = {mean,std} ({end-start} seconds)")
# -

header = "ndims,mode,dist_sigma,nsamples,c2st_score,crossvalid,total_cvtime_sec,nfolds,sklearn_version"
with open("rf_results.csv","w") as ocsv:
    ocsv.write(header+"\n")
    for k in samples.keys():
        for idx in range(len(sample_sizes)):
            scores = rf_results[k][idx]
            timings = rf_timings[k][idx]
            for cvid in range(len(timings)):
                score = scores[cvid]
                timing = timings[cvid]
                row = f"{NDIM},rf,{k},{sample_sizes[idx]},{score},{cvid},{timing},10,{sklversion}\n"
                ocsv.write(row)

# !head -n25 rf_results.csv|column -t -s","

# !tail -n25 rf_results.csv|column -t -s","
