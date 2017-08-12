#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains equation solvers. All the inputs and outputs are of type
:py:class:`~sympy.matrices.dense.MutableDenseMatrix`.
"""

from sympy import gcdex, Matrix, diag
from abelian.linalg.free_to_free import free_kernel
from abelian.linalg.utils import delete_zero_columns, remove_cols


def solve(A, b, p = None):
    """
    Solve the integer equation A * x = b mod p.

    The data (A, b, p) must be integer. The equation Ax = b mod p is solved,
    if a solution exists. If A is an epimorphism but not a monomorphism (i.e.
    overdetermined), one of the possible solutions is returned. If A is a
    monomorphism but not an epimorphism (i.e. underdetermined), a solution
    will be returned if one exists. If there is no solution, None is returned.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy integer matrix of size m x n.

    b : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy column matrix of size m x 1.

    p : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy column matrix of size m x 1. This column matrix represents
        the periods of the target group of A. If None, p will be set to the
        zero vector, i.e. infinite period in all components.

    Returns
    -------
    x : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A solution to A*x = b mod p, where x is of size n x 1.
        If no solution is found, None is returned.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> from abelian.linalg.utils import vector_mod_vector
    >>> A = Matrix([[5, 0, 3],
    ...             [0, 3, 4]])
    >>> x = Matrix([2, -1, 2])
    >>> p = Matrix([9, 9])
    >>> b = vector_mod_vector(A*x, p)
    >>> x_sol = solve(A, b, p)
    >>> vector_mod_vector(A*x_sol, p) == b
    True
    """

    # If no periods are supplied by the user, set the periods to zero,
    # i.e. infinite period or free-to-free.
    if p is None:
        m, n = b.shape
        p = Matrix(m, n, lambda i, j: 0)

    # Verify that the dimensions make sense
    (A_rows, A_cols) = A.shape
    (b_rows, b_cols) = b.shape
    (p_rows, p_cols) = p.shape
    if not (A_rows == b_rows == p_rows):
        raise ValueError('Dimension mismatch.')

    # Find the kernel of the projection onto the space Z_`p`
    ker_pi = delete_zero_columns(diag(*p))

    # Stack A | ker(pi) | b
    joined_A_D_b = A.row_join(ker_pi).row_join(b)

    # Compute ker( A | ker(pi) | b)
    kernel = free_kernel(joined_A_D_b)

    # The solution must be a linear combination of the columns of
    # ker( A | ker(pi) | b) such that the resulting vector has a -1
    # in the bottom entry.

    # Remove all columns with zero in the bottom entry
    m, n = kernel.shape
    col_indices = [j for j in range(n) if kernel[-1, j] == 0]
    kernel = remove_cols(kernel, col_indices)

    # Return None if the kernel is empty
    m, n = kernel.shape
    if n == 0:
        return None

    # Iteratively 'collapse' the columns using the extended
    # euclidean algorithm till the result is 1.
    m, n = kernel.shape
    while n > 1:
        # Compute the new column from the first two current ones
        f, g = kernel[-1, 0], kernel[-1, 1]
        (s, t, h) = gcdex(f, g)  # s*f + t*g = h.
        new_col = s * kernel[:, 0] + t * kernel[:, 1]

        # If there are only two columns, we have found the kernel
        if n == 2:
            kernel = new_col
            break

        # Delete current columns and insert the new one
        kernel = remove_cols(kernel, [0, 1])
        kernel = new_col.row_join(kernel)

        # Calculate new n value for the while-loop
        (m, n) = kernel.shape

    # Find shape of input, since shape of output depends on it
    (m, n) = A.shape

    # Make sure that the bottom row is -1 or 1.
    # It will always be 1 if the above while loop initiated,
    # but if it never initiated then value could be -1
    if kernel[-1, 0] not in [1, -1]:
        return None

    # The solution tot he problem is contained the first n rows of the
    # kernel, which is a column vector. Multiply by -1 if needed to
    # make sure the bottom entry is -1
    if kernel[-1, 0] == 1:
        return -kernel[:n, 0]
    else:
        return kernel[:n, 0]


def solve_epi(A, B, p = None):
    """
    Solve the equation X * mod p * A = B, where A is an epimorphism.

    The algorithm will produce a solution if (mod p * A) has a one
    sided inverse such that A_inv * A = I, i.e. A is an epimorphism.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy integer matrix of size m x n.

    B : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy column matrix of size k x n.

    p : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy column matrix of size m x 1. This column matrix represents
        the periods of the target group of A. If None, p will be set to the
        zero vector, i.e. infinite period.

    Returns
    -------
    x : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A solution to X * mod p * A = B.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> from abelian.linalg.utils import vector_mod_vector
    >>> A = Matrix([[5, 0, 3],
    ...             [0, 3, 4]])
    >>> X = Matrix([[1, 1],
    ...             [0, 1]])
    >>> B = X * A
    >>> X_sol = solve_epi(A, B)
    >>> X_sol * A == B
    True
    """
    # If no p (periods) are given, create a vector of zeros
    m, n = A.shape
    if p is None:
        p = Matrix(m, 1, lambda i, j : 0)

    # Verify the dimensions
    A_rows, A_cols = A.shape
    B_rows, B_cols = B.shape
    p_rows, p_cols = p.shape

    if (A_cols != B_cols) or (p.rows != A.rows):
        return ValueError('Dimension mismatch.')

    # Step 1: Find the inverse of A by solving A * x = e mod p
    # for each canonical generator e of the target space of A.
    # In other words, use the identity matrix to solve A * A_inv = I
    # by solving column-for-column.
    # Note: Only a one-sided inverse exists, A_inv * A = I is invalid.
    A_inv = []
    I_m = Matrix.eye(m)
    for i in range(m):
        identity_col = I_m[:, i]
        inv_col = solve(A, identity_col, p)

        # If a solution is found, append it
        if inv_col is not None:
            A_inv.append(inv_col)

    # Create the inverse of A from the list of column matrices
    A_inv = Matrix([[entry for entry in col] for col in A_inv]).T

    # Step 2: Multiply the inverse of A and B to form the solution
    # to the problem, the matrix X.
    # assert A * A_inv = I
    X = B * A_inv

    return X


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = True)



