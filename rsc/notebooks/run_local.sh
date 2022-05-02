#!/bin/sh

# nb=hyper.ipynb
# py=hyper.py

dst = ${1//.py/.ipynb}

# Ensure clean notebook. Purge and (re-)create. Only input cells now.
rm -fv ${dst}
poetry run jupytext --to notebook $1

# Pair if needed
##jupytext --set-formats ipynb,py:percent $nb

# Run all cells, save outputs
poetry run jupytext $dst --execute

# The same as `jupytext --execute` w/o jupytext
##jupyter nbconvert --to=notebook --inplace --ExecutePreprocessor.enabled=True $nb
