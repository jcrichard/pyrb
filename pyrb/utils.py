import numpy as np
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


def check_expected_return(mu):
    if np.isnan(mu).sum() > 0:
        raise ValueError('The expected returns vector contains missing values')


def check_risk_budget(riskbudgets, n):
    if np.isnan(riskbudgets).sum() > 0:
        raise ValueError('Risk budget contains missing values')
    if n != len(riskbudgets):
        raise ValueError('Risk budget size is not equal to the number of asset.')
    if np.isnan(riskbudgets < RISK_BUDGET_TOL).sum() > 0:
        raise ValueError(
            'One of the budget is smaller than {}. If you want a risk budget of 0 please remove the asset.'.format(
                RISK_BUDGET_TOL))
