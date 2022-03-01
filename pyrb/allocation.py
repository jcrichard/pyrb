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
        cov : array, shape (n, n)
            Covariance matrix of the returns.

        pi : array, shape(n,)
            Expected excess return for each asset (the default is None which implies 0 for each asset).

        x : array, shape(n,)
            Array  of weights.

        """
        self.__n = cov.shape[0]
        if x is None:
            x = np.array([np.nan] * self.n)
        self._x = x
        validation.check_covariance(cov)

        if pi is None:
            pi = np.array([0.0] * self.n)
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
        return (
            "solution x: {}\n"
            "lambda star: {}\n"
            "risk contributions: {}\n"
            "sigma(x): {}\n"
            "sum(x): {}\n"
        ).format(
            np.round(self.x * 100, 4),
            np.round(self.lambda_star * 100, 4),
            np.round(self.get_risk_contributions() * 100, 4),
            np.round(self.get_volatility() * 100, 4),
            np.round(self.x.sum() * 100, 4),
        )


class EqualRiskContribution(RiskBudgetAllocation):
    def __init__(self, cov):
        """
        Solve the equal risk contribution problem using cyclical coordinate descent. Although this does not change
        the optimal solution, the risk measure considered is the portfolio volatility.

        Parameters
        ----------
        cov : array, shape (n, n)
            Covariance matrix of the returns.

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
        cov : array, shape (n, n)
            Covariance matrix of the returns.

        budgets : array, shape(n,)
            Risk budgets for each asset (the default is None which implies equal risk budget).

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
        cov : array, shape (n, n)
            Covariance matrix of the returns.

        budgets : array, shape(n,)
            Risk budgets for each asset (the default is None which implies equal risk budget).

        pi : array, shape(n,)
            Expected excess return for each asset (the default is None which implies 0 for each asset).

        c : float
            Risk aversion parameter equals to one by default.
        """
        RiskBudgetAllocation.__init__(self, cov=cov, pi=pi)
        validation.check_risk_budget(budgets, self.n)
        self.budgets = budgets
        self.c = c

    def solve(self):
        x = solve_rb_ccd(cov=self.cov, budgets=self.budgets, pi=self.pi, c=self.c)
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
        return super().__str__() + "mu(x): {}\n".format(
            np.round(self.get_expected_return() * 100, 4)
        )


class ConstrainedRiskBudgeting(RiskBudgetingWithER):
    def __init__(
        self,
        cov,
        budgets=None,
        pi=None,
        c=1,
        C=None,
        d=None,
        bounds=None,
        solver="admm_ccd",
    ):
        """
        Solve the constrained risk budgeting problem. It supports linear inequality (Cx <= d) and bounds constraints.
        Notations follow the paper Constrained Risk Budgeting Portfolios by Richard J-C. and Roncalli T. (2019).

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

        solver : basestring
            "admm_ccd" (default): generalized standard deviation-based risk measure + linear constraints. The algorithm is ADMM_CCD (algorithm 4) and it solves equation (14).
            "admm_qp" : mean variance risk measure + linear constraints. The algorithm is ADMM_QP and it solves equation (15).

        """

        RiskBudgetingWithER.__init__(self, cov=cov, budgets=budgets, pi=pi, c=c)

        self.d = d
        self.C = C
        self.bounds = bounds
        validation.check_bounds(bounds, self.n)
        validation.check_constraints(C, d, self.n)
        self.solver = solver
        if (self.solver == "admm_qp") and (self.pi is not None):
            logging.warning(
                "The solver is set to 'admm_qp'. The risk measure is the mean variance in this case. The optimal "
                "solution will not be the same than 'admm_ccd' when pi is not zero.     "
            )

    def __str__(self):
        if self.C is not None:
            return (
                "solver: {}\n".format(self.solver)
                + "----------------------------\n"
                + super().__str__()
                + "C@x: {}\n".format(self.C @ self.x)
            )
        else:
            return super().__str__()

    def _sum_to_one_constraint(self, lamdba):
        x = self._lambda_solve(lamdba)
        sum_x = sum(x)
        return sum_x - 1

    def _lambda_solve(self, lamdba):
        if (
            self.C is None
        ):  # it is optimal to take the CCD in case of separable constraints
            x = solve_rb_ccd(
                self.cov, self.budgets, self.pi, self.c, self.bounds, lamdba
            )
            self.solver = "ccd"
        elif self.solver == "admm_qp":
            x = solve_rb_admm_qp(
                cov=self.cov,
                budgets=self.budgets,
                pi=self.pi,
                c=self.c,
                C=self.C,
                d=self.d,
                bounds=self.bounds,
                lambda_log=lamdba,
            )
        elif self.solver == "admm_ccd":
            x = solve_rb_admm_ccd(
                cov=self.cov,
                budgets=self.budgets,
                pi=self.pi,
                c=self.c,
                C=self.C,
                d=self.d,
                bounds=self.bounds,
                lambda_log=lamdba,
            )
        return x

    def solve(self):
        try:
            lambda_star = optimize.bisect(
                self._sum_to_one_constraint,
                0,
                BISECTION_UPPER_BOUND,
                maxiter=MAXITER_BISECTION,
            )
            self.lambda_star = lambda_star
            self._x = self._lambda_solve(lambda_star)
        except Exception as e:
            if e.args[0] == "f(a) and f(b) must have different signs":
                logging.exception(
                    "Bisection failed: "
                    + str(e)
                    + ". If you are using expected returns the parameter 'c' need to be correctly scaled (see remark 1 in the paper). Otherwise please check the constraints or increase the bisection upper bound."
                )
            else:
                logging.exception("Problem not solved: " + str(e))

    def get_risk_contributions(self, scale=True):
        """
        Return the risk contribution. If the solver is "admm_qp" the mean variance risk
        measure is considered.

        Parameters
        ----------
        scale : bool
            If True, the sum on risk contribution is scaled to one.

        Returns
        -------

        RC : array, shape (n,)
            Returns the risk contribution of each asset.

        """
        x = self.x
        cov = self.cov
        x = tools.to_column_matrix(x)
        cov = np.matrix(cov)

        if self.solver == "admm_qp":
            RC = np.multiply(x, cov * x) - self.c * self.x * self.pi
        else:
            RC = np.multiply(
                x, cov * x
            ).T / self.get_volatility() * self.c - tools.to_array(
                self.x.T
            ) * tools.to_array(
                self.pi
            )
        if scale:
            RC = RC / RC.sum()

        return tools.to_array(RC)
