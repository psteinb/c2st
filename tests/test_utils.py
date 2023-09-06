import numpy as np
from sklearn.model_selection import train_test_split

import pytest

from c2st.utils import train_test_split_idx
from c2st.utils import GaussianData


@pytest.mark.parametrize(
    "shuffle,stratify", [(True, None), (True, "y"), (False, None)]
)
def test_train_test_split_idx(shuffle, stratify):
    rng = np.random.default_rng()
    X = rng.random(size=(1000, 10))
    y = np.concatenate((np.zeros(500), np.ones(500)))
    seed = 123

    kwds = dict(
        test_size=0.2,
        shuffle=shuffle,
        random_state=seed,
        stratify=y if stratify == "y" else None,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, **kwds)
    idxs_train, idxs_test = train_test_split_idx(X, y, **kwds)

    assert (X_train == X[idxs_train]).all()
    assert (y_train == y[idxs_train]).all()
    assert (X_test == X[idxs_test]).all()
    assert (y_test == y[idxs_test]).all()
    
    
    
    
@pytest.fixture(scope='session')
def testdata():
    testdata = GaussianData(
            [10, 20, 30],
            np.linspace(0,2,9),
            np.linspace(0,2,9),
            10000
        )
    print("Data Generated")
    return testdata

def test_mean(testdata):
    print(testdata)
    for mean in testdata.means:
        for dim in testdata.NDIM:
            assert np.mean(testdata[dim][(mean,1.0)]) == pytest.approx(mean, rel=0.1, abs=0.1)
    
def test_var(testdata):
    for std_dev in testdata.std_devs:
        for dim in testdata.NDIM:
            assert np.var(testdata[dim][(0.0,std_dev)]) == pytest.approx(std_dev**2, rel=0.1)
    
def test_dim(testdata):
    for dim in testdata.NDIM:
        for mean in testdata.means:
            assert dim == testdata[dim][(0.0,1.0)].shape[1]
