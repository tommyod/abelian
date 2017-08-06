#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains equation solvers. All the inputs and outputs are of type
:py:class:`~sympy.matrices.dense.MutableDenseMatrix`.
"""

from sympy import gcdex, Matrix, diag
from abelian.linalg.free_to_free import free_kernel
from abelian.linalg.utils import delete_zero_columns, remove_cols




def solve_epi(A, b, target_periods):
    """
    Solve A*x = b, with x in Z^n and b in Z_`target_periods`.

    Long description.

    TODO Write this out.


    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy integer matrix of size m x n, A must be an epimorphism.

    b : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy column matrix of size m x 1.

    target_periods : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy column matrix of size m x 1.

    Returns
    -------
    x : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A solution to A*x = b, where x is of size n x 1.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[2, 0, 3],
    ...             [0, 1, 4]])
    >>> b = Matrix([5, 5])
    >>> x = Matrix([1, 1])
    >>> solve_epi(A, b, Matrix([0, 0])) == x
    True
    """

    # We are solving A * pi * x = b

    # Find the kernel of the projection onto the space Z_`target_periods`
    ker_pi = delete_zero_columns(diag(*target_periods))
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
    if kernel[-1, 0] != 1:
        raise ValueError('No solution found')

    # Find the solution as the first rows of the kernel, invert and return
    return kernel[:m, 0] * (-1)


def solve_eqnA(f1, f2, target_space):
    """
    Solve the matrix equation x \circ f_2 = f_1, where f_2 is epi.

    Long description.

    TODO Write this out.


    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy integer matrix of size m x n, A must be an epimorphism.

    b : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy column matrix of size m x 1.

    target_periods : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy column matrix of size m x 1.

    Returns
    -------
    x : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A solution to A*x = b, where x is of size n x 1.

    Examples
    ---------
    >>> 2 == 2
    True
    """

    # print('solve_eqnA({}, {})'.format(f1, f2))
    m, n = f2.shape
    k, n = f1.shape
    identity_m = Matrix.eye(m)

    # For each canonical generator
    for i in range(0, m):
        e = identity_m[:, i]

        # Create variable x on the first loop iteration
        if i == 0:
            # Attempt to solve. If not solveable, append a zero column
            try:
                x = f1 * solve_epi(f2, e, target_space)
            except ValueError:
                x = f1 * (e * 0)

        # Variable x is created, so row join on it x := x | new
        else:
            # Attempt to solve. If not solveable, append a zero column
            try:
                x = x.row_join(f1 * solve_epi(f2, e, target_space))
            except ValueError:
                x = x.row_join(f1 * (e * 0))

    # Delete zero columns and return
    return x  # delete_zero_columns(x)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = True)


if __name__ == "__main__":
    from random import randint as ri
    from sympy import pprint
    A = Matrix(3, 3, lambda i, j: ri(1, 9))
    pprint(A)
    x = Matrix([1, 1, 1])
    b = A*x
    pprint(solve_epi(A, b, target_periods=Matrix([0, 0, 0])))
    pprint(x)