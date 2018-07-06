import numpy as np
import quadprog
from .settings import *


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

def check_covariance(cov):
    if cov.shape[0] != cov.shape[1]:
        raise ValueError('The covariance matrix is not squared')
    if np.isnan(cov).sum().sum() > 0:
        raise ValueError('The covariance matrix contains missing values')


def check_expected_return(mu, n):
    if mu is None:
        return
    if n != len(mu):
        raise ValueError('Expected returns vector size is not equal to the number of asset.')
    if np.isnan(mu).sum() > 0:
        raise ValueError('The expected returns vector contains missing values')

def check_risk_budget(riskbudgets, n):
    if riskbudgets is None:
        return
    if np.isnan(riskbudgets).sum() > 0:
        raise ValueError('Risk budget contains missing values')
    if n != len(riskbudgets):
        raise ValueError('Risk budget size is not equal to the number of asset.')
    if all(v < RISK_BUDGET_TOL for v in riskbudgets):
        raise ValueError(
            'One of the budget is smaller than {}. If you want a risk budget of 0 please remove the asset.'.format(
                RISK_BUDGET_TOL))

def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None, bounds=None):
    n = P.shape[0]
    if bounds is not None:
        I = np.eye(n)
        LB = -I
        UB = I
    if G is None:
        G = np.vstack([LB, UB])
        h = np.array(np.hstack([-to_array(bounds[:, 0]), to_array(bounds[:, 1])]))
    else:
        G = np.vstack([G, LB, UB])
        h = np.array(np.hstack([h, -to_array(bounds[:, 0]), to_array(bounds[:, 1])]))
    qp_a = -q
    qp_G = P
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


def proximal_polyhedron(y, C, d, bound, A=None, b=None):
    n = len(y)
    return quadprog_solve_qp(np.eye(n), -np.array(y), C, np.array(d), A=A, b=b, bounds=bound)
