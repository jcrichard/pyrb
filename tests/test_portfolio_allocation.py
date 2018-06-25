import unittest
import numpy as np
from pyrb.allocation import EqualRiskContribution, RiskBudgeting


class AllocationTest(unittest.TestCase):
    def setUp(self):
        self.cov = np.matrix([[1, 0.1, -0.2], [0.1, 1, 0.3], [-0.2, 0.3, 1]])
        self.budgets = [0.2, 0.5, 0.3]
        self.ERC = EqualRiskContribution(self.cov)
        self.RB = RiskBudgeting(self.cov, self.budgets)


class PyrbTest(AllocationTest):
    def test_erc(self):
        self.ERC.solve()
        self.assertEqual(np.sum(self.ERC.x), 1)
        np.testing.assert_almost_equal(np.dot(np.dot(self.ERC.x, self.cov), self.ERC.x)[0, 0],
                                       self.ERC.get_risk_contributions(scale=False).sum(), decimal=10)
        self.assertTrue(abs(self.ERC.get_risk_contributions().mean() - 1.0 / 3) < 1e-5)

    def test_rb(self):
        self.RB.solve()
        self.assertEqual(np.sum(self.RB.x), 1)
        np.testing.assert_almost_equal(np.dot(np.dot(self.RB.x, self.cov), self.RB.x)[0, 0],
                                       self.RB.get_risk_contributions(scale=False).sum(), decimal=10)
        self.assertTrue(abs(self.RB.get_risk_contributions() - self.budgets).sum() < 1e-5)
