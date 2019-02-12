Constrained and Unconstrained Risk budgeting allocation in Python
================

This repository contains the code for solving risk budgeting
with generalized standard deviation-based risk measure:

R(x) = - pi^T x + c ( x^T Sigma x)¨0.5

This formulation encompasses Gaussian value-at-risk and Gaussian expected shortfall.

The algorithm is efficient for large dimension and suitable for backtesting.
A description can be found in [*A Fast Algorithm for Computing High-Dimensional Risk Parity Portfolios*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2325255)
by Théophile Griveau-Billion, Jean-Charles Richard and Thierry Roncalli and [*Constrained Risk Budgeting Portfolios*]()
by Jean-Charles Richard and Thierry Roncalli.

You can solve
------------------

- Equally risk contribution
- Risk budgeting
- Risk parity with expected return
- Constrained Risk parity

Installation
------------------
 Can be done using ``pip``: 

    pip install git+https://github.com/jcrichard/pyrb


Usage
------------------

    from pyrb import EqualRiskContribution

    ERC = EqualRiskContribution(cov)
    ERC.solve()
    ERC.get_risk_contribution()
    ERC.get_volatility()


References
------------------

>Griveau-Billion, T., Richard, J-C., and Roncalli, T. (2013), A Fast Algorithm for Computing High-dimensional Risk Parity Portfolios, SSRN.

>Maillard, S., Roncalli, T. and
    Teiletche, J. (2010), The Properties of Equally Weighted Risk Contribution Portfolios,
    Journal of Portfolio Management, 36(4), pp. 60-70.
    
>Richard, J-C., and Roncalli, T. (2015), Smart
    Beta: Managing Diversification of Minimum Variance Portfolios, in Jurczenko, E. (Ed.),
    Risk-based and Factor Investing, ISTE Press -- Elsevier.
    
>Roncalli, T. (2015), Introducing Expected Returns into Risk Parity Portfolios: A New Framework for Asset Allocation,
    Bankers, Markets & Investors, 138, pp. 18-28.
 
