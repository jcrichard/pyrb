import unittest

import numpy as np

from pyrb.allocation import EqualRiskContribution, RiskBudgeting, ConstrainedRiskBudgeting


class AllocationTest(unittest.TestCase):
    def setUp(self):
        self.cor = np.array([[1, 0.1, 0.4, 0.5, 0.5], [0.1, 1, 0.7, 0.4, 0.4], [
            0.4, 0.7, 1, 0.8, 0.05], [0.5, 0.4, 0.8, 1, 0.1], [0.5, 0.4, 0.05, 0.1, 1]])
        self.vol = [0.15, 0.20, 0.25, 0.3, 0.1]
        self.n = len(self.vol)
        self.cov = self.cor * np.outer(self.vol, self.vol)
        self.budgets = [0.2, 0.2, 0.3, 0.1, 0.2]
        self.bounds = np.array(
            [[0.2, 0.3], [0.2, 0.3], [0.05, 0.15], [0.05, 0.15], [0.25, 0.35]])
        self.ERC = EqualRiskContribution(self.cov)
        self.RB = RiskBudgeting(self.cov, self.budgets)
        self.CRB = ConstrainedRiskBudgeting(
            self.cov, budgets=None, pi=None, bounds=self.bounds)


class PyrbTest(AllocationTest):
    def test_erc(self):
        self.ERC.solve()
        np.testing.assert_almost_equal(np.sum(self.ERC.x), 1)
        np.testing.assert_almost_equal(
            np.dot(
                np.dot(
                    self.ERC.x,
                    self.cov),
                self.ERC.x) ** 0.5,
            self.ERC.get_risk_contributions(
                scale=False).sum(),
            decimal=10)
        self.assertTrue(
            abs(self.ERC.get_risk_contributions().mean() - 1.0 / self.n) < 1e-5)

    def test_rb(self):
        self.RB.solve()
        np.testing.assert_almost_equal(np.sum(self.RB.x), 1, decimal=5)
        np.testing.assert_almost_equal(
            np.dot(
                np.dot(
                    self.RB.x,
                    self.cov),
                self.RB.x) ** 0.5,
            self.RB.get_risk_contributions(
                scale=False).sum(),
            decimal=10)
        self.assertTrue(
            abs(self.RB.get_risk_contributions() - self.budgets).sum() < 1e-5)

    def test_cerb(self):
        self.CRB.solve()
        np.testing.assert_almost_equal(np.sum(self.CRB.x), 1)
        np.testing.assert_almost_equal(
            self.CRB.get_risk_contributions()[1], 0.2455, decimal=5)
        np.testing.assert_almost_equal(np.sum(self.CRB.x[1]), 0.2)


if __name__ == "__main__":
    unittest.main()
