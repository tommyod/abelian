#!/usr/bin/env python
# -*- coding: utf-8 -*-


from sympy import Matrix, gcdex, pprint
from abelian.utils import mod

def hermite_normal_form(A):
    """
    Compute U and H such that A*U = H.

    This algorithm computes the column version of the
    Hermite normal form [1]_, and returns a tuple (U, H)
    such that A*U = H. The matrix U is an unimodular transformation
    matrix and H is the result of the transformation, i.e.
    H is in Hermite normal form.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The matrix to factor.

    Returns
    -------
    U : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        An unimodular matrix.

    H : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A matrix in Hermite normal form.


    References
    ----------
    .. [1] Derek F. Holt. Handbook of Computational Group Theory.
          Discrete Mathematics and Its Applications. Boca Raton,
          Fla: Chapman & Hall/CRC, 2005.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2],
    ...             [3, 4]])
    >>> U, H = hermite_normal_form(A)
    >>> U.det() in [1, -1] # Determinant is +/- 1
    True
    >>> A*U == H
    True
    """

    # Get size and set up matrices U and H
    m, n = A.shape
    j = 0
    H = A.copy()
    U = Matrix.eye(n)

    # Iterate down the rows of the matrix
    for i in range(0, m):

        # If every entry to the right is a zero, no pivot will be found
        # for this row and we move on to the next one.
        if H[i, j:] == H[i, j:] * 0:
            continue

        # Create zeros to the right of the pivot H[i, j]
        for k in range(j + 1, n):

            # Skip the column if the element is zero
            # In this case the column index j in not incremented
            if H[i, k] == 0:
                continue

            # Apply the 'elementary hermite transform' to the columns of H,
            # ignoring the top i rows of H as they are identically zero
            # The implementation of the 'elementary hermite transform'
            # does not explicitly compute matrix products.
            # Equivalent to right-multiplication by Matrix([[a, -s/g],
            #                                               [b, r/g]])

            # Extended Euclidean algorithm, so that r*a + s*b = g = gcd(r, s)
            r, s = H[i, j], H[i, k]
            a, b, g = gcdex(r, s)

            # Apply the matrix product of H and U
            H[i:, j], H[i:, k] = (a * H[i:, j] + b * H[i:, k],
                                  -(s / g) * H[i:, j] + (r / g) * H[i:, k])
            U[:, j], U[:, k] = (a * U[:, j] + b * U[:, k],
                                -(s / g) * U[:, j] + (r / g) * U[:, k])

        # Make sure the pivot element is positive.
        # Some saving achieved by realizing that the first i rows of H are
        # identically zero -- thus no multiplication is needed.
        if H[i, j] < 0:
            H[i:, j] = -H[i:, j]
            U[:, j] = -U[:, j]

        # Making all elements to the left of the pivot H[i, j] smaller
        # than the pivot and positive using division algorithm transform
        for k in range(0, j):
            # Compute quotient in the division algorithm, subtracting
            # the quotient times H[:, j] leaves a positive remainder
            a = H[i, k] // H[i, j]
            H[:, k] = H[:, k] - a * H[:, j]
            U[:, k] = U[:, k] - a * U[:, j]
        # Increment j (the column index). Break if j is out of dimension
        j += 1
        if j >= n:
            # j is out of dimension, break
            break

    # assert A * U == H
    return U, H

def smith_normal_form(A, compute_transformation=True):
    """
    Compute (U, S, V) such that U*A*V = S.

    This algorithm computes the Smith normal form [1]_
    of an integer matrix. If `compute_transformation`
    is True, it returns matrices U, S, V such that
    U * A * V = S, where U and V are unimodular and S
    is in Smith normal form. If `compute_transformation`
    is false, it returns S and does not compute U and V.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The matrix to factor.
    compute_transformation : bool
        If True, computes and returns (U, S, V). If, false, returns S.

    Returns
    -------
    U : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        An unimodular matrix.

    S : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A matrix in Smith normal form.

    V : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        An unimodular matrix.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2],
    ...             [3, 4]])
    >>> U, S, V = smith_normal_form(A)
    >>> U.det() in [1, -1] and V.det() in [1, -1] # Unimodularity
    True
    >>> U * A * V == S # Verify factorization
    True
    >>> K = smith_normal_form(A, compute_transformation=False)
    >>> K == S
    True
    """

    # Get size and set up the unimodular matrices U and V
    m, n = A.shape
    min_m_n = min(m, n)
    S = A[:, :]
    if compute_transformation:
        U, V = Matrix.eye(m), Matrix.eye(n)

    def row_col_all_zero(matrix, f):
        """
        Check that all entries to the right of and below `f` are zero.
        """
        for entry in matrix[f, f + 1:]:
            if entry != 0:
                return False
        for entry in matrix[f + 1:, f]:
            if entry != 0:
                return False
        return True

    # Main loop, iterate over all sub-matrices to reduce
    f = 0
    while f < min_m_n:

        # While there are non-zero elements to reduce in row/column f
        # and the diagonal element is not positive
        while not (row_col_all_zero(S, f) and S[f, f] >= 0):

            # Find index pair of minimum non-zero entry (in absolute value)
            # in the sub-matrix S[f:, f:].
            indices = ((i, j) for j in range(f, n) for i in range(f, m))
            key_val_pairs = ((index, abs(S[index])) for index in indices
                             if abs(S[index]) != 0)
            (i, j), min_val = min(key_val_pairs, key=lambda k: k[1])

            # Permute S to move the minimal element to the pivot location
            S[f:, j], S[f:, f] = S[f:, f], S[f:, j]
            S[i, f:], S[f, f:] = S[f, f:], S[i, f:]
            if compute_transformation:
                V[:, j], V[:, f] = V[:, f], V[:, j]
                U[i, :], U[f, :] = U[f, :], U[i, :]

            # If the freshly permuted pivot is negative, make it positive
            if S[f, f] < 0:
                S[f:, f] = -S[f:, f]
                if compute_transformation:
                    V[:, f] = -V[:, f]

            # Reduce row f so every entry is smaller than pivot
            for k in range(f + 1, n):
                if S[f, k] == 0:
                    continue
                # Subtract a times column f from column k
                a = S[f, k] // S[f, f]
                S[f:, k] = S[f:, k] - a * S[f:, f]
                if compute_transformation:
                    V[:, k] = V[:, k] - a * V[:, f]

            # Reduce column f so every entry is smaller than pivot
            for k in range(f + 1, m):
                if S[k, f] == 0:
                    continue
                # Subtract a times row f from row k
                a = S[k, f] // S[f, f]
                S[k, f:] = S[k, f:] - a * S[f, f:]
                if compute_transformation:
                    U[k, :] = U[k, :] - a * U[f, :]

        f += 1

    # Enforce divisibility criterion using the 'divisibility transformation'
    # matrices.
    for f in range(min_m_n):
        for k in range(f + 1, min_m_n):

            # Divisibility criterion is fulfilled
            if mod(S[k, k], S[f, f]) == 0:
                continue

            # S[f, f] does not divide S[k, k]
            r, s = S[f, f], S[k, k]
            a, b, c = gcdex(r, s)
            S[f, f], S[k, k] = c, (r * s) // c
            if compute_transformation:
                V[:, f], V[:, k] = V[:, f] + V[:, k], -b * (s / c) * V[:,
                                                                     f] + a * (
                                   r / c) * V[:, k]
                U[f, :], U[k, :] = a * U[f, :] + b * U[k, :], -(s / c) * U[f,
                                                                         :] + (
                                   r / c) * U[k, :]

    if compute_transformation:
        return U, S, V
    else:
        return S



if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = True)

