# c2st

## two sample test using a ML classifier

``` python
> from c2st.check import c2st as compare

#let's say you have 2 samples X and Y and want to know 
#if they come from the same distribution

> compare(X,Y)
[0.51904464]
```

The `compare` method returns (by default) the accuracy on how well a binary classifyer was able to classify Y from X while being trained on the concatenated dataset `(X,Y)`. All samples of `X` have received the label `0` and `Y` has received the label `1`.

## in `rsc`

Small validation study which NN architecture exposes more utility for `c2st` two sample testing.

# References

`c2st` is a sample based method to evaluate good-of-fit based on two ensembles only. For more details, see 

- Friedman, J. "On Multivariate Goodness-of-Fit and Two-Sample Testing", https://www.osti.gov/biblio/826696/

- Lopez-Paz et al, "Revisiting Classifier Two-Sample Tests", http://arxiv.org/abs/1610.06545
