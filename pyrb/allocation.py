import logging
from abc import abstractmethod

import numpy as np
import scipy.optimize as optimize

from . import tools
from . import validation
from .settings import BISECTION_UPPER_BOUND, MAXITER_BISECTION
from .solvers import solve_rb_ccd, solve_rb_admm_qp, solve_rb_admm_ccd


class RiskBudgetAllocation:

    @property
    def cov(self):
        return self.__cov

    @property
    def x(self):
        return self._x

    @property
    def pi(self):
        return self.__pi

    @property
    def n(self):
        return self.__n

    def __init__(self, cov, pi=None, x=None):
        """
        Base class for Risk Budgeting Allocation.

        Parameters
        ----------
        cov: asset covariance matrix.
        pi: asset expected excess returns. None by default
        x: asset allocation.
        """
        self.__n = cov.shape[0]
        if x is None:
            x = np.array([np.nan] * self.n)
        self._x = x
        validation.check_covariance(cov)

        if pi is None:
            pi = np.array([0] * self.n)
        validation.check_expected_return(pi, self.n)
        self.__pi = tools.to_column_matrix(pi)

        self.__cov = np.array(cov)
        self.lambda_star = np.nan

    @abstractmethod
    def solve(self):
        """Solve the problem."""
        pass

    @abstractmethod
    def get_risk_contributions(self):
        """Get the risk contribution of the Risk Budgeting Allocation."""
        pass

    def get_variance(self):
        """Get the portfolio variance: x.T * cov * x."""
        x = self.x
        cov = self.cov
        x = tools.to_column_matrix(x)
        cov = np.matrix(cov)
        RC = np.multiply(x, cov * x)
        return np.sum(tools.to_array(RC))

    def get_volatility(self):
        """Get the portfolio volatility: x.T * cov * x."""
        return self.get_variance() ** 0.5

    def get_expected_return(self):
        """Get the portfolio expected excess returns: x.T * pi."""
        if self.pi is None:
            return np.nan
        else:
            x = self.x
            x = tools.to_column_matrix(x)
        return float(x.T * self.pi)

    def __str__(self):
        return ('solution x: {}\n'
                'lambda star: {}\n'
                'risk contributions: {}\n'
                'sigma(x): {}\n'
                'sum(x): {}\n'
                ).format(np.round(self.x * 100, 4),
                         np.round(self.lambda_star * 100, 4),
                         np.round(self.get_risk_contributions() * 100, 4),
                         np.round(self.get_volatility() * 100, 4),
                         np.round(self.x.sum() * 100, 4))


class EqualRiskContribution(RiskBudgetAllocation):
    def __init__(self, cov):
        """
        Solve the equal risk contribution problem using cyclical coordinate descent. Although this does not change
        the optimal solution, the risk measure considered is the portfolio volatility.

        Parameters
        ----------
        cov: The covariance matrix
        """

        RiskBudgetAllocation.__init__(self, cov)

    def solve(self):
        x = solve_rb_ccd(cov=self.cov)
        self._x = tools.to_array(x / x.sum())
        self.lambda_star = self.get_volatility()

    def get_risk_contributions(self, scale=True):
        x = self.x
        cov = self.cov
        x = tools.to_column_matrix(x)
        cov = np.matrix(cov)
        RC = np.multiply(x, cov * x) / self.get_volatility()
        if scale:
            RC = RC / RC.sum()
        return tools.to_array(RC)


class RiskBudgeting(RiskBudgetAllocation):

    def __init__(self, cov, budgets):
        """
        Solve the risk budgeting problem using cyclical coordinate descent. Although this does not change
        the optimal solution, the risk measure considered is the portfolio volatility.

        Parameters
        ----------
        cov: the covariance matrix.
        budgets: array of risk budget.
        """
        RiskBudgetAllocation.__init__(self, cov=cov)
        validation.check_risk_budget(budgets, self.n)
        self.budgets = budgets

    def solve(self):
        x = solve_rb_ccd(cov=self.cov, budgets=self.budgets)
        self._x = tools.to_array(x / x.sum())
        self.lambda_star = self.get_volatility()

    def get_risk_contributions(self, scale=True):
        x = self.x
        cov = self.cov
        x = tools.to_column_matrix(x)
        cov = np.matrix(cov)
        RC = np.multiply(x, cov * x) / self.get_volatility()
        if scale:
            RC = RC / RC.sum()
        return tools.to_array(RC)


class RiskBudgetingWithER(RiskBudgetAllocation):

    def __init__(self, cov, budgets=None, pi=None, c=1):
        """
        Solve the risk budgeting problem for the standard deviation risk measure using cyclical coordinate descent.
        The risk measure is given by R(x) = c * sqrt(x^T cov x) -  pi^T x.

        Parameters
        ----------
        cov: the covariance matrix.
        budgets: array of risk budget.
        pi: array of expected excess return.
        c: risk aversion paramter.
        """
        RiskBudgetAllocation.__init__(self, cov=cov, pi=pi)
        validation.check_risk_budget(budgets, self.n)
        self.budgets = budgets
        self.c = c

    def solve(self):
        x = solve_rb_ccd(cov=self.cov, budgets=self.budgets, pi=self.pi)
        self._x = tools.to_array(x / x.sum())
        self.lambda_star = -self.get_expected_return() + self.get_volatility() * self.c

    def get_risk_contributions(self, scale=True):
        x = self.x
        cov = self.cov
        x = tools.to_column_matrix(x)
        cov = np.matrix(cov)
        RC = np.multiply(x, cov * x) / self.get_volatility() * self.c - self.x * self.pi
        if scale:
            RC = RC / RC.sum()
        return tools.to_array(RC)

    def __str__(self):
        return super().__str__() + \
               "mu(x): {}\n".format(np.round(self.get_expected_return() * 100, 4))


class ConstrainedRiskBudgeting(RiskBudgetingWithER):
    def __init__(self, cov, budgets, pi, c=1, C=None, d=None, bounds=None, solver="admm_ccd"):
        RiskBudgetingWithER.__init__(
            self, cov=cov, budgets=budgets, pi=pi, c=c)

        self.d = d
        self.C = C
        self.bounds = bounds
        validation.check_bounds(bounds, self.n)
        validation.check_constraints(C, d, self.n)
        self.solver = solver

    def __str__(self):
        if self.C is not None:
            return super().__str__() + \
                   "C@x: {}\n".format(self.C @ self.x)
        else:
            return super().__str__()

    def _sum_to_one_constraint(self, lamdba):
        x = self._lambda_solve(lamdba)
        sum_x = sum(x)
        return sum_x - 1

    def _lambda_solve(self, lamdba):
        if self.C is None:  # it is optimal to take the CCD in case of separable constraints
            x = solve_rb_ccd(self.cov, self.budgets, self.pi, self.c, self.bounds, lamdba)
            self.solver = "ccd"
        elif self.solver == "admm_qp":
            x = solve_rb_admm_qp(self.cov, self.budgets, self.pi, self.c, self.C, self.d, self.bounds, lamdba)
        elif self.solver == "admm_ccd":
            x = solve_rb_admm_ccd(self.cov, self.budgets, self.pi, self.c, self.C, self.d, self.bounds, lamdba)
        return x

    def solve(self):
        try:
            lambda_star = optimize.bisect(self._sum_to_one_constraint, 0, BISECTION_UPPER_BOUND,
                                          maxiter=MAXITER_BISECTION)
            self.lambda_star = lambda_star
            self._x = self._lambda_solve(lambda_star)
        except Exception as e:
            if e.args[0] == 'f(a) and f(b) must have different signs':
                logging.exception(
                    "Bisection failed: " +
                    str(e) +
                    ". If you are using expected returns the parameter 'c' need to be correctly scaled (see remark 1 in the paper). Otherwise please check the constraints or increase the bisection upper bound.")
            else:
                logging.exception("Problem not solved: " + str(e))

    def get_risk_contributions(self, scale=True):
        x = self.x
        cov = self.cov
        x = tools.to_column_matrix(x)
        cov = np.matrix(cov)

        if self.solver == "admm_qp":
            RC = np.multiply(x, cov * x) * self.c - self.x * self.pi
        else:
            RC = np.multiply(x, cov * x).T / self.get_volatility() * self.c - tools.to_array(self.x.T) * tools.to_array(
                self.pi)
        if scale:
            RC = RC / RC.sum()
        return tools.to_array(RC)
