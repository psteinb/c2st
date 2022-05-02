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


from __future__ import annotations

import time
from functools import partial

import numpy as np
from c2st.check import c2st_
from numpy.random import default_rng
from sklearn import __version__ as sklversion
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier

FIXEDSEED = 1328


# -

NDIM = 10
max_nsamples = 2048
sample_sizes = [2**it for it in range(7, 9)]
sample_sizes.append(max_nsamples)
RNG = default_rng(FIXEDSEED)
print(sample_sizes)

# +
center_normal = partial(RNG.multivariate_normal, mean=np.zeros(NDIM), cov=np.eye(NDIM))
distributions = {0.0: center_normal}

for alpha in np.linspace(0, 2, 9):
    distributions[alpha] = partial(
        RNG.multivariate_normal, mean=np.zeros(NDIM) + alpha, cov=np.eye(NDIM)
    )


# +
center_samples = center_normal(size=max_nsamples)
samples = {}
for k, v in distributions.items():
    samples[k] = v(size=max_nsamples)

assert samples[0.0].shape == (max_nsamples, 10)

# +
rf_results = {}
rf_timings = {}
total = len(samples.values()) * len(sample_sizes)
cnt = 0


for k, v in samples.items():
    for size in sample_sizes:

        try:
            start = time.time()
            scores = c2st_(X=center_samples[:size, ...], Y=v[:size, ...], n_folds=10)
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
        rf_timings[k].append(len(scores) * [end - start])
        cnt += 1

        print(f"{cnt}/{total}: {k}[{size},...] = {mean,std} ({end-start} seconds)")
# -

header = "ndims,mode,dist_sigma,nsamples,c2st_score,crossvalid,total_cvtime_sec,nfolds,sklearn_version"
with open("rf_results.csv", "w") as ocsv:
    ocsv.write(header + "\n")
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

# +
import pandas as pd

df = pd.read_csv("rf_results.csv")
df.c2st_score.plot()
