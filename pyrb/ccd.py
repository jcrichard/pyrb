import numpy as np
from . import utils
from .settings import *
import logging


def solve_rb_ccd(cov, budget = None, mu = None, c=None, C = None, d = None, bounds = None, lambda_log = 1):
    """

    :param cov:
    :param budget:
    :param mu:
    :param c:
    :param C:
    :param d:
    :param bounds:
    :param lamdba:
    :return: The vector of weights solved with CCD.
    """

    Sigma = np.array(cov)
    n = np.shape(Sigma)[0]

    if c is None:
        c = 0
        mu = np.zeros((n, 1))

    if budget is None:
        budget = np.ones((n, 1)) / n

    if bounds is None:
        bounds = np.matrix([[RISK_BUDGET_TOL] * n, [1] * n]).T

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
            gamma_ = -lambda_log * budget[i]

            x_tilde = (-beta + np.sqrt(beta ** 2 - 4 * alpha * gamma_)) / (2 * alpha)
            x[i] = x_tilde

            if (C is not None) or (bounds is not None):
                pi = utils.proximal_polyhedron(utils.to_array(x), C, d, bounds)
                x[i] = pi[i]

            Sx = np.matmul(Sigma, x)
        cvg = np.sum((x / np.sum(x) - x0 / np.sum(x0)) ** 2) <= COVERGENCE_TOL
        x0 = x.copy()
        iters = iters + 1
        if iters >= MAX_ITER:
            logging.info("Maximum iteration reached: {}".format(MAX_ITER))
            break
    return utils.to_array(x)
