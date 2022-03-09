import numpy as np

from .settings import RISK_BUDGET_TOL


def check_covariance(cov):
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("The covariance matrix is not squared")
    if np.isnan(cov).sum().sum() > 0:
        raise ValueError("The covariance matrix contains missing values")


def check_expected_return(mu, n):
    if mu is None:
        return
    if n != len(mu):
        raise ValueError(
            "Expected returns vector size is not equal to the number of asset."
        )
    if np.isnan(mu).sum() > 0:
        raise ValueError("The expected returns vector contains missing values")


def check_constraints(C, d, n):
    if C is None:
        return
    if n != C.shape[1]:
        raise ValueError("Number of columns of C is not equal to the number of asset.")
    if len(d) != C.shape[0]:
        raise ValueError("Number of rows of C is not equal to the length  of d.")


def check_bounds(bounds, n):
    if bounds is None:
        return
    if n != bounds.shape[0]:
        raise ValueError(
            "The number of rows of the bounds array is not equal to the number of asset."
        )
    if 2 != bounds.shape[1]:
        raise ValueError(
            "The number of columns the bounds array should be equal to two (min and max bounds)."
        )


def check_risk_budget(riskbudgets, n):
    if riskbudgets is None:
        return
    if np.isnan(riskbudgets).sum() > 0:
        raise ValueError("Risk budget contains missing values")
    if (np.array(riskbudgets) < 0).sum() > 0:
        raise ValueError("Risk budget contains negative values")
    if n != len(riskbudgets):
        raise ValueError("Risk budget size is not equal to the number of asset.")
    if all(v < RISK_BUDGET_TOL for v in riskbudgets):
        raise ValueError(
            "One of the budget is smaller than {}. If you want a risk budget of 0 please remove the asset.".format(
                RISK_BUDGET_TOL
            )
        )
