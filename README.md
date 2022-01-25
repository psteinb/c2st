# c2st

## two sample test using a ML classifier

``` python
from c2st.check import c2st as compare

#let's say you have 2 samples X and Y and want to know 
#if they come from the same distribution

probs = compare(X,Y)
print(probs)
```

## in `rsc`

Small validation study which NN architecture exposes more utility for `c2st` two sample testing.

# References

`c2st` is a sample based method to evaluate good-of-fit based on two ensembles only. For more details, see 

- Friedman, J. "On Multivariate Goodness-of-Fit and Two-Sample Testing", https://www.osti.gov/biblio/826696/

- Lopez-Paz et al, "Revisiting Classifier Two-Sample Tests", http://arxiv.org/abs/1610.06545
