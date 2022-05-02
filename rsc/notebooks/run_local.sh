#!/bin/sh

# nb=hyper.ipynb
# py=hyper.py

# Ensure clean notebook. Purge and (re-)create. Only input cells now.
rm -f $2
jupytext --to notebook $1

# Pair if needed
##jupytext --set-formats ipynb,py:percent $nb

# Run all cells, save outputs
##jupytext $nb --execute

# The same as `jupytext --execute` w/o jupytext
##jupyter nbconvert --to=notebook --inplace --ExecutePreprocessor.enabled=True $nb
