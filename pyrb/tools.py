import quadprog

import numpy as np


def to_column_matrix(x):
    """Return x as a matrix columns."""
    x = np.matrix(x)
    if x.shape[1] != 1:
        x = x.T
    if x.shape[1] == 1:
        return x
    else:
        raise ValueError("x is not a vector")


def to_array(x):
    """Turn a columns or row matrix to an array."""
    if x is None:
        return None
    elif (len(x.shape)) == 1:
        return x

    if x.shape[1] != 1:
        x = x.T
    return np.squeeze(np.asarray(x))


def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None, bounds=None):
    """Quadprog helper."""
    n = P.shape[0]
    if bounds is not None:
        I = np.eye(n)
        LB = -I
        UB = I
        if G is None:
            G = np.vstack([LB, UB])
            h = np.array(np.hstack([-to_array(bounds[:, 0]), to_array(bounds[:, 1])]))
        else:
            G = np.vstack([G, LB, UB])
            h = np.array(
                np.hstack([h, -to_array(bounds[:, 0]), to_array(bounds[:, 1])])
            )

    qp_a = q  # because  1/2 x^T G x - a^T x
    qp_G = P
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraints
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


def proximal_polyhedra(y, C, d, bound, A=None, b=None):
    """Wrapper for projecting a vector on the constrained set."""
    n = len(y)
    return quadprog_solve_qp(
        np.eye(n), np.array(y), np.array(C), np.array(d), A=A, b=b, bounds=bound
    )
