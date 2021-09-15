import logging

import numba
import numpy as np

from . import tools
from .settings import CCD_COVERGENCE_TOL, MAX_ITER, MAX_WEIGHT, ADMM_TOL


@numba.njit
def accelarate(_varphi, r, s, u, alpha=10, tau=2):
    """
    Update varphy and dual error for accelerating convergence after ADMM steps.

    Parameters
    ----------
    _varphi
    r: primal_error.
    s: dual error.
    u: primal_error.
    alpha: error treshld.
    tau: scaling parameter.

    Returns
    -------
    updated varphy and primal_error.
    """

    primal_error = np.sum(r ** 2)
    dual_error = np.sum(s * s)
    if primal_error > alpha * dual_error:
        _varphi = _varphi * tau
        u = u / tau
    elif dual_error > alpha * primal_error:
        _varphi = _varphi / tau
        u = u * tau
    return _varphi, u


@numba.jit('Tuple((float64[:], float64[:], float64))(float64[:], float64, float64[:], float64, float64, float64[:], float64[:], int32[:], float64[:,:], float64, float64[:,:])',
      nopython=True)
def _cycle(x, c, var, _varphi, sigma_x, Sx, budgets, pi, bounds, lambda_log, cov):
    """
    Internal numba function for computing one cycle of the CCD algorithm.

    """
    n = len(x)
    for i in range(n):
        alpha = c * var[i] + _varphi * sigma_x
        beta = c * (Sx[i] - x[i] * var[i]) - pi[i] * sigma_x
        gamma_ = -lambda_log * budgets[i] * sigma_x
        x_tilde = (-beta + np.sqrt(beta ** 2 - 4 * alpha * gamma_)) / (2 * alpha)

        x_tilde = np.maximum(np.minimum(x_tilde, bounds[i, 1]), bounds[i, 0])

        x[i] = x_tilde
        Sx = np.dot(cov, x)
        sigma_x = np.sqrt(np.dot(Sx, x))
    return x, Sx, sigma_x


def solve_rb_ccd(
    cov, budgets=None, pi=None, c=1.0, bounds=None, lambda_log=1.0, _varphi=0.0
):
    """
    Solve the risk budgeting problem for standard deviation risk-based measure with bounds constraints using cyclical
    coordinate descent (CCD). It is corresponding to solve equation (17) in the paper.

    By default the function solve the ERC portfolio or the RB portfolio if budgets are given.

    Parameters
    ----------
    cov : array, shape (n, n)
        Covariance matrix of the returns.

    budgets : array, shape (n,)
        Risk budgets for each asset (the default is None which implies equal risk budget).

    pi : array, shape (n,)
        Expected excess return for each asset (the default is None which implies 0 for each asset).

    c : float
        Risk aversion parameter equals to one by default.

    bounds : array, shape (n, 2)
        Array of minimum and maximum bounds. If None the default bounds are [0,1].

    lambda_log : float
        Log penalty parameter.

    _varphi : float
        This parameters is only useful for solving ADMM-CCD algorithm should be zeros otherwise.

    Returns
    -------
    x : aray shape(n,)
        The array of optimal solution.

    """

    n = cov.shape[0]

    if bounds is None:
        bounds = np.zeros((n, 2))
        bounds[:, 1] = MAX_WEIGHT
    else:
        bounds = np.array(bounds * 1.0)

    if budgets is None:
        budgets = np.array([1] * n) / n
    else:
        budgets = np.array(budgets)
    budgets = budgets / np.sum(budgets)

    if (c is None) | (pi is None):
        c = 1.0
        pi = np.array([0] * n)
    else:
        c = float(c)
        pi = np.array(pi).astype(float)

    # initial value equals to 1/vol portfolio
    x = 1 / np.diag(cov) ** 0.5 / (np.sum(1 / np.diag(cov) ** 0.5))
    x0 = x / 100

    budgets = tools.to_array(budgets)
    pi = tools.to_array(pi)
    var = np.array(np.diag(cov))
    Sx = tools.to_array(np.dot(cov, x))
    sigma_x = np.sqrt(np.dot(Sx, x))

    cvg = False
    iters = 0

    while not cvg:
        x, Sx, sigma_x = _cycle(
            x, c, var, _varphi, sigma_x, Sx, budgets, pi, bounds, lambda_log, cov
        )
        cvg = np.sum(np.array(x - x0) ** 2) <= CCD_COVERGENCE_TOL
        x0 = x.copy()
        iters = iters + 1
        if iters >= MAX_ITER:
            logging.info(
                "Maximum iteration reached during the CCD descent: {}".format(MAX_ITER)
            )
            break

    return tools.to_array(x)


def solve_rb_admm_qp(
    cov,
    budgets=None,
    pi=None,
    c=None,
    C=None,
    d=None,
    bounds=None,
    lambda_log=1,
    _varphi=1,
):
    """
    Solve the constrained risk budgeting constraint for the Mean Variance risk measure:
    The risk measure is given by R(x) =  x^T cov x - c * pi^T x

    Parameters
    ----------
    cov : array, shape (n, n)
        Covariance matrix of the returns.

    budgets : array, shape (n,)
        Risk budgets for each asset (the default is None which implies equal risk budget).

    pi : array, shape (n,)
        Expected excess return for each asset (the default is None which implies 0 for each asset).

    c : float
        Risk aversion parameter equals to one by default.

    C : array, shape (p, n)
        Array of p inequality constraints. If None the problem is unconstrained and solved using CCD
        (algorithm 3) and it solves equation (17).

    d : array, shape (p,)
        Array of p constraints that matches the inequalities.

    bounds : array, shape (n, 2)
        Array of minimum and maximum bounds. If None the default bounds are [0,1].

    lambda_log : float
        Log penalty parameter.

    _varphi : float
        This parameters is only useful for solving ADMM-CCD algorithm should be zeros otherwise.

    Returns
    -------
    x : aray shape(n,)
        The array of optimal solution.

    """

    def proximal_log(a, b, c, budgets):
        delta = b * b - 4 * a * c * budgets
        x = (b + np.sqrt(delta)) / (2 * a)
        return x

    cov = np.array(cov)
    n = np.shape(cov)[0]

    if bounds is None:
        bounds = np.zeros((n, 2))
        bounds[:, 1] = MAX_WEIGHT
    else:
        bounds = np.array(bounds * 1.0)

    if budgets is None:
        budgets = np.array([1.0 / n] * n)

    x0 = 1 / np.diag(cov) / (np.sum(1 / np.diag(cov)))

    x = x0 / 100
    z = x.copy()
    zprev = z
    u = np.zeros(len(x))
    cvg = False
    iters = 0
    pi_vec = tools.to_array(pi)
    identity_matrix = np.identity(n)

    while not cvg:

        # x-update
        x = tools.quadprog_solve_qp(
            cov + _varphi * identity_matrix,
            c * pi_vec + _varphi * (z - u),
            G=C,
            h=d,
            bounds=bounds,
        )

        # z-update
        z = proximal_log(_varphi, (x + u) * _varphi, -lambda_log, budgets)

        # u-update
        r = x - z
        s = _varphi * (z - zprev)
        u += x - z

        # convergence check
        cvg1 = sum((x - x0) ** 2)
        cvg2 = sum((x - z) ** 2)
        cvg3 = sum((z - zprev) ** 2)
        cvg = np.max([cvg1, cvg2, cvg3]) <= ADMM_TOL
        x0 = x.copy()
        zprev = z

        iters = iters + 1
        if iters >= MAX_ITER:
            logging.info("Maximum iteration reached: {}".format(MAX_ITER))
            break

        # parameters update
        _varphi, u = accelarate(_varphi, r, s, u)

    return tools.to_array(x)


def solve_rb_admm_ccd(
    cov,
    budgets=None,
    pi=None,
    c=None,
    C=None,
    d=None,
    bounds=None,
    lambda_log=1,
    _varphi=1,
):
    """
    Solve the constrained risk budgeting constraint for the standard deviation risk measure:
    The risk measure is given by R(x) = c * sqrt(x^T cov x) -  pi^T x

    Parameters
    ----------
    Parameters
    ----------
    cov : array, shape (n, n)
        Covariance matrix of the returns.

    budgets : array, shape (n,)
        Risk budgets for each asset (the default is None which implies equal risk budget).

    pi : array, shape (n,)
        Expected excess return for each asset (the default is None which implies 0 for each asset).

    c : float
        Risk aversion parameter equals to one by default.

    C : array, shape (p, n)
        Array of p inequality constraints. If None the problem is unconstrained and solved using CCD
        (algorithm 3) and it solves equation (17).

    d : array, shape (p,)
        Array of p constraints that matches the inequalities.

    bounds : array, shape (n, 2)
        Array of minimum and maximum bounds. If None the default bounds are [0,1].

    lambda_log : float
        Log penalty parameter.

    _varphi : float
        This parameters is only useful for solving ADMM-CCD algorithm should be zeros otherwise.

    Returns
    -------
    x : aray shape(n,)
        The array of optimal solution.


    """

    cov = np.array(cov)

    x0 = 1 / np.diag(cov) / (np.sum(1 / np.diag(cov)))

    x = x0 / 100
    z = x.copy()
    zprev = z
    u = np.zeros(len(x))
    cvg = False
    iters = 0
    pi_vec = tools.to_array(pi)
    while not cvg:

        # x-update
        x = solve_rb_ccd(
            cov,
            budgets=budgets,
            pi=pi_vec + (_varphi * (z - u)),
            bounds=bounds,
            lambda_log=lambda_log,
            c=c,
            _varphi=_varphi,
        )

        # z-update
        z = tools.proximal_polyhedra(x + u, C, d, A=None, b=None, bound=bounds)

        # u-update
        r = x - z
        s = _varphi * (z - zprev)
        u += x - z

        # convergence check
        cvg1 = sum((x - x0) ** 2)
        cvg2 = sum((x - z) ** 2)
        cvg3 = sum((z - zprev) ** 2)
        cvg = np.max([cvg1, cvg2, cvg3]) <= ADMM_TOL
        x0 = x.copy()
        zprev = z

        iters = iters + 1
        if iters >= MAX_ITER:
            logging.info("Maximum iteration reached: {}".format(MAX_ITER))
            break

        # parameters update
        _varphi, u = accelarate(_varphi, r, s, u)

    return tools.to_array(x)
