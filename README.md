# c2st

## Two sample test using a ML classifier

Test whether sets of D-dimensional points are samples from the same
multivariate probability distribution.

The `c2st` function returns the accuracy of how well a binary classifier was
able to classify two sets of points `X` and `Y` while being trained on the
concatenated dataset `(X,Y)`. All samples of `X` have received the label `0`
and `Y` has received the label `1`.

A value close to 0.5 means that the classifier is not better than random
guessing, i.e. `X` and `Y` are likely from the same distribution. A value close
to 1 means the classifier was able to separate `X` and `Y`, so they are
probably samples from different distributions.


```py
>>> import numpy as np
>>> from c2st.check import c2st

>>> rng=np.random.default_rng(seed=123)

# same distribution (Gaussian N(0,1)), D=20, 1000 points each
>>> X=rng.normal(loc=0, scale=1, size=(1000,20))
>>> Y=rng.normal(loc=0, scale=1, size=(1000,20))
>>> c2st(X, Y)
0.4970122828225085

# now shift the mean of Y by 0.3
>>> Y=rng.normal(loc=0.3, scale=1, size=(1000,20))
0.6964673015530594

# let's move Y more extensively away from X to 1.5
>>> Y=rng.normal(loc=1.5, scale=1, size=(1000,20))
>>> c2st(X, Y)
0.9994791666666666

# or change the distribution's variance, so we compare a narrow to a wide distribution at the same loc
>>> Y=rng.normal(loc=0, scale=2, size=(1000,20))
>>> c2st(X, Y)
0.950321845047345

# we use balanced accuracy scoring by default to handle different sample sizes
>>> Y=rng.normal(loc=0, scale=1, size=(300,20))
>>> c2st(X, Y)
0.5013659591194969
```

## In `rsc`

Small validation study which NN architecture exposes more utility for `c2st`
two sample testing. At this point, the analysis you find in
[rsc/c2st_results.ipynb](rsc/c2st_results.ipynb) has by far not any academic
scrutiny as you'd find in Lopez-Paz et al. However, it can serve as guidance
where which implementation of `c2st` can shine.
If you'd like to redo the analysis, consult the notebook provided in [rsc/c2st_quality_study.ipynb](rsc/c2st_quality_study.ipynb).


# References

`c2st` is a sample based method to evaluate goodness-of-fit based on two ensembles only. For more details, see

- Friedman, J. "On Multivariate Goodness-of-Fit and Two-Sample Testing", https://www.osti.gov/biblio/826696/

- Lopez-Paz et al, "Revisiting Classifier Two-Sample Tests", http://arxiv.org/abs/1610.06545
