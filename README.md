Risk budgeting allocation in Python
================

This repository contains the code for solving risk budgeting problems for large universe. The algorithm is decribed in the paper  [*A Fast Algorithm for Computing High-Dimensional Risk Parity Portfolios*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2325255) 
by Théophile Griveau-Billion, Jean-Charles Richard and Thierry Roncalli. 

You can solve
------------------

- Equally risk contribution
- Risk budgeting
- Risk parity with expected return


Usage
------------------
Since PyMC3 Models is built on top of scikit-learn, you can use the same methods as with a scikit-learn model.

    from pyrb import EqualRiskContribution()

    ERC = EqualRiskContribution()
    ERC.solve(cov)
    ERC.get_risk_contribution()
    ERC.get_vol()


References
------------------


License
------------------
`Apache License, Version 2.0 < >`__
