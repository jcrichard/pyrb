import numpy as np
from . import utils
from abc import abstractmethod
from .settings import *
from .ccd import solve_rb_ccd


class Allocation(object):
    """

    Simple allocation object.

    """

    @property
    def cov(self):
        return self.__cov

    @property
    def mu(self):
        return self.__mu

    @property
    def x(self):
        return self._x

    @property
    def n(self):
        return self.__n

    def __init__(self, cov, mu=None, x=None):
        self.__n = cov.shape[0]
        self._x = x
        utils.check_covariance(cov)
        if mu is None:
            self.__mu = np.matrix(np.zeros(self.n)).T
        else:
            utils.check_expected_return(mu)
            self.__mu = utils.to_column_matrix(mu)
        self.__cov = np.matrix(cov)

    @abstractmethod
    def solve(self):
        pass

    def get_risk_contributions(self, scale=True):
        """

        Args:
            scale: If true the risk contribution sum to 100% otherwise it sums to the portfolio variance.

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


class EqualRiskContribution(Allocation):
    """
    Solve the equal risk contribution problem using cyclical coordinate descent.

    Args:
        cov: The covariance matrix

    """

    def __init__(self, cov):
        Allocation.__init__(self, cov)

    def solve(self):
        x = solve_rb_ccd(cov=self.cov)
        self._x = utils.to_array(x / x.sum())


class RiskBudgeting(Allocation):
    def __init__(self, cov, riskbudget):
        Allocation.__init__(self, cov)

        utils.check_risk_budget(riskbudget, self.n)
        self.riskbudget = riskbudget

    def solve(self):
        x = solve_rb_ccd(cov=self.cov, budget=self.riskbudget)
        self._x = utils.to_array(x / x.sum())
