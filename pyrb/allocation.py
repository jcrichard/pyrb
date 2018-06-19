__all__ = ['EqualRiskContribution', 'RiskBudgeting']

import numpy as np
from . import utils
from abc import abstractmethod
from  tolerance import *


class Portfolio():
    def __init__(self, cov, mu=None):
        self.n = cov.shape[0]
        self.x = None
        if mu is None:
            self.mu = np.matrix(np.zeros(self.n)).T
        else:
            self.mu = np.matrix(mu).T
        self.mu = utils.to_column_matrix(self.mu)
        self.cov = np.matrix(cov)
        if self.cov.shape[0] != self.cov.shape[1]:
            raise ValueError('The covariance is not squared')
        if np.isnan(self.cov).sum().sum() > 0:
            raise ValueError('cov contains missing values')
        if np.isnan(self.mu).sum() > 0:
            raise ValueError('mu contains missing values')

    @abstractmethod
    def solve(self):
        pass

    def get_risk_contributions(self, scale=True):
        """

        Args:
            scale: If true the risk contribution sum to 100% otherwise it is portfolio variance.

        Returns: The risk contribution of the portfolio: x .* (Cov * x).

        """
        x = self.x
        cov = self.cov
        x = utils.to_column_matrix(x)
        cov = np.matrix(cov)
        RC = np.multiply(x, cov * x)
        if scale:
            RC = RC / RC.sum()
        return utils.to_array(RC)

    def get_variance(self):
        """

        Returns: The portfolio variance: x.T * Cov * x.

        """
        x = self.x
        cov = self.cov
        x = utils.to_column_matrix(x)
        cov = np.matrix(cov)
        RC = np.multiply(x, cov * x)
        return np.sum(utils.to_array(RC))

    def get_volatility(self):
        """

        Returns: The portfolio volatility: sqrt(x.T * Cov * x.)

        """
        return self.get_variance() ** 0.5

    def get_expected_return(self):
        """

        Returns: The portfolio expected return: x.T * mu

        """
        if self.mu is None:
            return np.nan
        else:
            x = self.x
            x = utils.to_column_matrix(x)
        return x.T * self.mu


class EqualRiskContribution(Portfolio):
    """
    Solve the equal risk contribution problem using cyclical coordinate descent.

    Args:
        cov: The covariance matrix

    """

    def __init__(self, cov):
        Portfolio.__init__(self, cov, None)

    def solve(self):
        Sigma = np.array(self.cov)
        n = np.shape(Sigma)[0]
        x0 = np.ones((n, 1)) / n
        x = x0 * 0
        var = np.diag(Sigma)
        Sx = np.matmul(Sigma, x)
        cvg = False

        iters = 0
        while not cvg:
            for i in range(n):
                alpha = var[i];
                beta = (Sx[i] - x[i] * var[i])[0]
                gamma_ = -1.0 / n;
                x_tilde = (-beta + np.sqrt(beta ** 2 - 4 * alpha * gamma_)) / (2 * alpha);
                x[i] = x_tilde;
                Sx = np.matmul(Sigma, x)
            cvg = np.sum((x / np.sum(x) - x0 / np.sum(x0)) ** 2) <= TOL;

            x0 = x.copy()
            iters = iters + 1
            if iters >= MAXITER:
                self.cvg = "Maximum iteration reached."
                break

        self.x = utils.to_array(x / x.sum())


class RiskBudgeting(Portfolio):
    def __init__(self, cov, riskbudget):
        Portfolio.__init__(self, cov, None)
        self.riskbudget = riskbudget

        if np.isnan(self.riskbudget).sum() > 0:
            raise ValueError('riskbudget contains missing values')
        if self.n != len(self.riskbudget):
            raise ValueError('riskbudget size is not equal to the number of asset.')
        if np.isnan(riskbudget < BUDGETTOL).sum() > 0:
            raise ValueError('One of the budget is smaller than {}'.format(BUDGETTOL))

    def solve(self):
        Sigma = np.array(self.cov)
        n = np.shape(Sigma)[0]
        x0 = np.ones((n, 1)) / n
        x = x0 * 0
        var = np.diag(Sigma)
        Sx = np.matmul(Sigma, x)
        cvg = False

        iters = 0
        while not cvg:
            for i in range(n):
                alpha = var[i];
                beta = (Sx[i] - x[i] * var[i])[0]
                gamma_ = -1.0 * self.riskbudget[i];
                x_tilde = (-beta + np.sqrt(beta ** 2 - 4 * alpha * gamma_)) / (2 * alpha);
                x[i] = x_tilde;
                Sx = np.matmul(Sigma, x)
            cvg = np.sum((x / np.sum(x) - x0 / np.sum(x0)) ** 2) <= TOL;

            x0 = x.copy()
            iters = iters + 1
            if iters >= MAXITER:
                self.cvg = "Maximum iteration reached."
                break

        self.x = np.squeeze(x / x.sum())
