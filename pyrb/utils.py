import numpy as np


def to_column_matrix(x):
    x = np.matrix(x)
    if x.shape[1] != 1:
        x = x.T
    if x.shape[1] == 1:
        return x
    else:
        raise ValueError("x is not a vector")


def to_array(x):
    return np.squeeze(np.asarray(x))
