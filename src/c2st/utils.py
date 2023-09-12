
import numpy as np
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

import collections
from numpy.random import default_rng
from torch.utils.data import Dataset
def train_test_split_idx(
    X, y, test_size=0.2, shuffle=True, stratify=None, random_state=None
):
    """Implements a subset of sklearn.model_selection.train_test_split() but
    only returns train and test array indices instead of data copies to save
    memory.
    """
    assert X.ndim == 2
    assert y.ndim == 1
    assert test_size > 0, "test_size must be > 0"
    n_samples = X.shape[0]
    assert n_samples == len(y), "X and y must have equal length"
    n_test = int(test_size * n_samples)
    assert n_test > 0, f"{n_test=}, increase test_size"
    n_train = n_samples - n_test

    if not shuffle:
        idxs_train = np.arange(n_train)
        idxs_test = np.arange(n_train, n_train + n_test)
    else:
        if stratify is not None:
            CVClass = StratifiedShuffleSplit
        else:
            CVClass = ShuffleSplit

        cv = CVClass(
            test_size=n_test, train_size=n_train, random_state=random_state
        )

        idxs_train, idxs_test = next(cv.split(X=X, y=stratify))
    return idxs_train, idxs_test


def print_csv(
    filename: str,
    nlines: int,
    head: bool = True
) -> None:
    """
    OS-agnostic function to print first (or last) N lines from a csv file
    """
    with open(filename, 'r') as input_file:
        lines = input_file.readlines()
    rows = [line.split(',') for line in lines]
    if head:
        rows = rows[:nlines+1]
    else:
        rows = rows[-nlines-1:]
    widths = [max(map(len, col)) for col in zip(*rows)]
    for row in rows:
        print("  ".join((val.ljust(width) for val, width in zip(row, widths))))
        
        
    

class GaussianData(collections.abc.Sequence):
    def __init__(self, NDIM, means, std_devs, max_nsamples, seed=420):
        self.NDIM = NDIM
        self.means = means
        self.std_devs = std_devs
        self.max_nsamples = max_nsamples
        self.data = [dict() for _ in range(len(NDIM))]
        self.center_samples = dict()
        RNG = default_rng(seed)
        for i, dim in enumerate(NDIM):
            self.center_samples[dim] = RNG.multivariate_normal(
                mean=np.zeros(dim), 
                cov=np.eye(dim),
                size=max_nsamples
            )
            for mean in means:
                for std_dev in std_devs:
                    self.data[i][(mean, std_dev)] = RNG.multivariate_normal(
                        mean=np.zeros(dim)+mean, 
                        cov=np.eye(dim) * (std_dev**2),
                        size=max_nsamples
                    )
                        
    def __getitem__(self, index):
        return self.data[self.NDIM.index(index)]
    def __len__(self):
        return len(self.data)
    
    
def modify_dataset(
    dataset: Dataset, 
    class_zero: list, 
    class_one: list
) -> Dataset:
    """
    Modify a Dataset instance to mark a set of classes with 1.0 and others with 0.0
    If the class is found in both class zero and class one, Then it's taken fully but if the 
    class is found only in either class zero or class one, only half of it is taken.
    e.g. -> This will assign 1.0 to half and 0.0 to the other half randomly
        X = [0,1,2,3,4]
        Y = [0,1,2,3,4]
        dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), train=True)
        dataset = modify_dataset(dataset, X, Y)
    """
    one = class_one
    zero = class_zero
    def label_transform(label):
        if label in one and label in zero:
            return np.random.choice([0.0, 1.0])
        elif label in one:
            return 1.0
        return 0.0
    dataset.target_transform = label_transform
    indices = torch.rand(dataset.targets.size()) > 1.0
    for num in range(0,10):
        if num in one and num in zero:
            indices |= (dataset.targets == num)
        elif num in one or num in zero:
            indices |= (dataset.targets == num) & (torch.rand(indices.size()) > 0.5)

    dataset.data, dataset.targets = dataset.data[indices], dataset.targets[indices] 
    
    return dataset  