import numpy as np
from . import utils
from .settings import *
import logging


def solve_rb_ccd(cov, budget=None, mu=None, c=None, bounds=None):
    """

    Args:
        cov:
        budget:
        mu:
        c:
        bounds:

    Returns:

    """

    Sigma = np.array(cov)
    n = np.shape(Sigma)[0]
    lamdba = 1

    if c is None:
        c = 0
        mu = np.zeros((n, 1))

    if budget is None:
        budget = np.ones((n, 1)) / n

    x0 = np.ones((n, 1)) / n
    x = x0 * 0
    var = np.diag(Sigma)
    Sx = np.matmul(Sigma, x)
    cvg = False
    iters = 0
    while not cvg:
        for i in range(n):
            alpha = var[i]
            beta = (Sx[i] - x[i] * var[i])[0] - c * mu[i]
            gamma_ = -lamdba * budget[i]
            if bounds is None:
                x_tilde = (-beta + np.sqrt(beta ** 2 - 4 * alpha * gamma_)) / (2 * alpha)
            else:
                x_tilde = np.minimum(np.maximum((-beta + np.sqrt(beta ** 2 - 4 * alpha * gamma_)) / (2 * alpha),
                                                bounds[i, 0]), bounds[i, 1])
            x[i] = x_tilde
            Sx = np.matmul(Sigma, x)
        cvg = np.sum((x / np.sum(x) - x0 / np.sum(x0)) ** 2) <= COVERGENCE_TOL
        x0 = x.copy()
        iters = iters + 1
        if iters >= MAX_ITER:
            logging.info("Maximum iteration reached: {}".format(MAX_ITER))
            break
    return utils.to_array(x)
