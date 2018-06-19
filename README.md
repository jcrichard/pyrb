Risk budgeting allocation in Python
================

This repository contains the code for solving risk budgeting problems on large universe. The algorithm is decribed in the paper  [*A Fast Algorithm for Computing High-Dimensional Risk Parity Portfolios*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2325255) 
by Théophile Griveau-Billion, Jean-Charles Richard and Thierry Roncalli. 

You can solve
------------------

- Equally risk contribution
- Risk budgeting
- Risk parity with expected return


Usage
------------------

    from pyrb import EqualRiskContribution

    ERC = EqualRiskContribution()
    ERC.solve(cov)
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
    
License
------------------
 
