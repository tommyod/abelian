#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains a set of utility functions which are used by the
other modules in the `linalg` package. The functions defined herein
operate on matrices, or are at the very least related to linear algebra
computations.
"""

import functools
from functools import partial
from sympy import Matrix, gcd, lcm
from abelian.utils import mod


def columns_as_list(A):
    """
    Returns the columns of A as a list of lists.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy matrix.

    Returns
    -------
    list_of_cols : list
        A list of lists, where each sub_list is a column,
        e.g. structure [[col1], [col2], ...].

    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2],
    ...             [3, 4]])
    >>> list_of_cols = columns_as_list(A)
    >>> list_of_cols
    [[1, 3], [2, 4]]
    """
    m, n = A.shape
    list_of_cols = [[A[i, j] for i in range(0, m)] for j in range(0, n)]
    return list_of_cols


def nonzero_columns(H):
    """
    Counts the number of columns in H not identically zero.

    Parameters
    ----------
    H : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy matrix.

    Returns
    -------
    nonzero_cols : int
        The number of columns of A not indentically zero.

    Examples
    ---------
    >>> from sympy import Matrix, diag
    >>> A = Matrix([[0, 2],
    ...             [0, 4]])
    >>> nonzero_columns(A)
    1
    >>> nonzero_columns(Matrix.eye(5))
    5
    >>> nonzero_columns(diag(0,1,0,3,5,0))
    3
    """
    m, n = H.shape
    nonzero_cols = 0
    # Loop over the columns
    for j in range(0, n):

        # Loop over the rows
        for i in range(0, m):
            if H[i, j] != 0:
                nonzero_cols += 1
                break

    return nonzero_cols


def diagonal_rank(S):
    """
    Count the number of non-zero diagonals in S,
    where S is in Smith normal form.

    Parameters
    ----------
    S : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy matrix in Smith normal form.

    Returns
    -------
    num_nonzeros : int
        The number of non-zeros on the diagonal.

    Examples
    ---------
    >>> from sympy import diag
    >>> diagonal_rank(diag(1,2,0,0,0,0))
    2
    >>> diagonal_rank(diag(1,2,4,8,0,0))
    4
    """
    return len(nonzero_diag_as_list(S))


def nonzero_diag_as_list(S):
    """
    Return a list of the non-zero diagonals entries of S.

    Parameters
    ----------
    S : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy matrix, typically in Smith normal form.

    Returns
    -------
    nonzero_diags : list
        A list of the non-zero diagonal entries of S.

    Examples
    ---------
    >>> from sympy import diag
    >>> nonzero_diag_as_list(diag(1,2,0,0,0,0))
    [1, 2]
    >>> nonzero_diag_as_list(diag(1,2,4,8,0,0))
    [1, 2, 4, 8]
    """
    m, n = S.shape
    nonzero_diags = [S[i, i] for i in range(0, min(m, n)) if S[i, i] != 0]
    return nonzero_diags


def remove_zero_columns(M):
    """
    Return a copy of M where the columns that are identically zero are deleted.

    Parameters
    ----------
    M : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy matrix with zero or more columns which are identically zero.

    Returns
    -------
    M : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A copy of the input matrix with all zero columns removed.

    Examples
    ---------
    >>> from sympy import Matrix, diag
    >>> A = Matrix([[0, 1],
    ...             [0, 2]])
    >>> remove_zero_columns(A) == Matrix([1, 2])
    True
    >>> A = diag(0,1,2)
    >>> A_del = Matrix([[0, 0],
    ...                 [1, 0],
    ...                 [0, 2]])
    >>> remove_zero_columns(A) == A_del
    True
    """

    A = M.copy()
    m, n = A.shape

    # STEP 1: Find the columns to remove
    cols_to_remove = []

    # Iterate over columns
    for j in range(0, n):
        all_zero = True
        for i in range(0, m):
            if A[i, j] != 0:
                all_zero = False
                break

        # If the column has all zeros, we delete it later
        if all_zero:
            cols_to_remove.append(j)

    # STEP 2: Remove the columns and return
    return remove_cols(A, cols_to_remove)


def remove_cols(A, cols_to_remove):
    """
    Return a copy of A where the columns with indices in `cols_to_remove`
    are removed.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy matrix.

    cols_to_remove : list
        A list of column indices to remove from `A`.

    Returns
    -------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A copy of the input matrix with removed columns.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[5, 6, 7, 8]])
    >>> B = remove_cols(A, [0, 2])
    >>> B == Matrix([[6, 8]])
    True
    """
    m, n = A.shape
    if any([index >= n for index in cols_to_remove]):
        raise ValueError('Index to remove not in matrix.')

    cols_to_remove.sort()

    new_A = A.copy()
    deleted = 0
    for j in cols_to_remove:
        new_A.col_del(j - deleted)
        deleted += 1

    return new_A


def remove_rows(A, rows_to_remove):
    """
    Return a copy of A where the rows with indices in `rows_to_remove`
    are removed.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy matrix.

    rows_to_remove : list
        A list of row indices to remove from `A`.

    Returns
    -------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A copy of the input matrix with removed rows.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[5, 6, 7, 8]]).T
    >>> B = remove_rows(A, [0, 2])
    >>> B == Matrix([[6, 8]]).T
    True
    """
    return remove_cols(A.T, rows_to_remove).T

def vector_mod_vector(vector, mod_vector):
    """
    Return `vector` % `mod_vector`, a vectorized mod operation.

    Parameters
    ----------
    vector : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy column vector, i.e. a sympy matrix of dimension m x 1.
    mod_vector : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy column vector, i.e. a sympy matrix of dimension m x 1.

    Returns
    -------
    modded_vector : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The result of the mod operation on every entry.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> element = Matrix([5, 7, 9])
    >>> mod_vect = Matrix([3, 3, 5])
    >>> modded = vector_mod_vector(element, mod_vect)
    >>> modded == Matrix([2, 1, 4])
    True
    """

    if len(vector) != len(mod_vector):
        raise ValueError('Vectors must be of same length.')

    entries = (mod(i, j) for (i,j) in zip(vector, mod_vector))
    modded_vector = Matrix(list(entries))
    return modded_vector


def matrix_mod_vector(A, mod_col):
    """
    Returns a copy of `A` with every column modded by `mod_col`.

    Parameters
    ----------
    vector : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy matrix of size m x n.
    mod_col : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy column vector, i.e. a sympy matrix of dimension m x 1.

    Returns
    -------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A copy of the input with each column modded.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[5, 6],
    ...             [8, 5],
    ...             [3, 5]])
    >>> mod_col = Matrix([4, 6, 3])
    >>> A_modded = matrix_mod_vector(A, mod_col)
    >>> A_modded == Matrix([[1, 2],
    ...                     [2, 5], [0, 2]])
    True
    """
    m, n = A.shape
    if m != len(mod_col):
        raise ValueError('Dimension mismatch.')

    new_A = A.copy()

    # If it's a zero matrix (point),
    # return immediately.
    if m * n == 0:
        return new_A

    for j in range(0, n):
        column = A[:, j]
        new_A[:, j] = vector_mod_vector(column, mod_col)

    return new_A


def order_of_vector(v, mod_vector):
    """
    Returns the order of the element `v` in a FGA like `mod_vector`.

    Parameters
    ----------
    v : :py:class:`~sympy.matrices.dense.MutableDenseMatrix` or a list
        An iterable object with integers. This is the group element.
    mod_vector : :py:class:`~sympy.matrices.dense.MutableDenseMatrix` or a list
        An iterable object with integers. This is the orders of the group.

    Returns
    -------
    order : int
        The order of `v` in `mod_vector`.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> order_of_vector([1,2,3], [2,4,6]) # Order of 2
    2
    >>> order_of_vector([1,2,3], [0, 0, 0]) # Order of 0 (infinite order)
    0
    >>> order_of_vector([1,2,3], [7, 5, 2]) # lcm(7, 10, 2) is 70
    70
    >>> order_of_vector([0,0,0], [0,0,0]) # Identity element
    1
    >>> order_of_vector([0,2, 3], [0,0,0]) # Non-trivial element
    0
    >>> order_of_vector([1, 0, 1], [5, 0, 0])
    0

    """

    # Arguments must have the same length
    if len(v) != len(mod_vector):
        raise ValueError('The arguments must have the same length.')

    def div(top, bottom):
        # Division, but if we divide by zero we define it as unity
        try:
            return top // bottom
        except:
            return 1

    # Create the generate and iterate through it
    gen = zip(v, mod_vector)
    gcd_list = (div(order, gcd(element, order)) for(element, order) in gen)

    # Reduce over the iterator with the least common multiple
    return functools.reduce(lcm, gcd_list)


def mat_times_diag(A, diagonal):
    """
    Multiply a dense matrix and a diagonal.

    Multiplies `A` with a column vector `diagonal`, which is interpreted as
    the diagonal of a matrix. This algorithm exploids the diagonal structure
    to reduce the number of computations.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A dense matrix.
    diag : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The diagonal of a matrix, represented as a sympy column vector.

    Returns
    -------
    product : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The product `A` times `diag`.

    Examples
    ---------
    >>> from sympy import Matrix, diag
    >>> A = Matrix([[1, 2],
    ...             [3, 4]])
    >>> diagonal = Matrix([2, 3])
    >>> mat_times_diag(A, diagonal) == A * diag(2, 3)
    True
    """
    m, n = A.shape
    new_A = A.copy()
    for col in range(0, n):
        new_A[:, col] *= diagonal[col]
    return new_A


def diag_times_mat(diagonal, A):
    """
    Multiply a diagonal and a dense matrix.

    Multiplies a column vector `diagonal` with `A`, in that order.
    This algorithm exploids the diagonal structure
    to reduce the number of computations.

    Parameters
    ----------
    diag : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The diagonal of a matrix, represented as a sympy column vector.
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A dense matrix.

    Returns
    -------
    product : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The product `diag` times `A`.

    Examples
    ---------
    >>> from sympy import Matrix, diag
    >>> A = Matrix([[1, 2],
    ...             [3, 4]])
    >>> diagonal = Matrix([2, 3])
    >>> diag_times_mat(diagonal, A) == diag(2, 3) * A
    True
    """

    return mat_times_diag(A.T, diagonal).T


def reciprocal_entrywise(A):
    """
    Returns the entrywise reciprocal of a matrix or vector.

    Will skip zero entries.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy matrix, or vector ( m x 1 matrix).

    Returns
    -------
    reciprocal : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The entrywise reciprocal of `A`.

    Examples
    ---------
    >>> from sympy import Matrix, diag
    >>> D = diag(1, 2, 3)
    >>> D_inv = reciprocal_entrywise(D)
    >>> D * D_inv == Matrix.eye(3)
    True
    >>> A = Matrix([[1, 5], [4, 1]])
    >>> A_recip = reciprocal_entrywise(A)
    >>> A_recip == Matrix([[1, 1/5], [1/4, 1]])
    True
    """
    m, n = A.shape
    def recip(x):
        if x == 0:
            return x
        return 1/x

    return Matrix(m, n, [recip(e) for e in A])


def norm(vector, p = 2):
    """
    The p-norm of an iterable.

    Parameters
    ----------
    vector : :py:class:`~sympy.matrices.dense.MutableDenseMatrix` or list
        The iterable to compute the norm over.
    p : float
        The p-value in the p-norm. Should be between 1 and infinity (None).

    Returns
    -------
    norm : float
        The computed norm.

    Examples
    --------
    >>> vector = [1, 2, 3]
    >>> norm(vector, 1)
    6.0
    >>> norm(tuple(vector), None)
    3.0
    >>> norm(iter(vector), None)
    3.0
    >>> norm(vector, None)
    3.0
    >>> norm(vector, 2)
    3.7416573867739413
    >>> from sympy import Matrix
    >>> vector = Matrix(vector)
    >>> norm(vector, 1)
    6.0
    >>> norm(vector, None)
    3.0
    >>> norm(vector, 2)
    3.7416573867739413

    """

    # If no p is given, assume the infinity norm and return the max value
    if p == None:
        return float(max(abs(j) for j in list(vector)))

    # P should not be smaller than 1.
    if p < 1:
        raise ValueError('p must be >= 1 for the p-norm.')

    # Return the p-norm
    norm = sum(abs(i)**p for i in vector)**(1/p)
    return float(norm)


def difference(iterable1, iterable2, p = None):
    """
    Compute the difference with a p-norm.

    Parameters
    ----------
    iterable1 : :py:class:`~sympy.matrices.dense.MutableDenseMatrix` or list
        The iterable to compute the norm over.
    iterable2 : :py:class:`~sympy.matrices.dense.MutableDenseMatrix` or list
        The iterable to compute the norm over.
    p : float
        The p-value in the p-norm. Should be between 1 and infinity (None).

    Returns
    -------
    norm : float
        The computed norm of the difference.

    Examples
    --------
    >>> 2 + 2
    4

    """
    vector1 = Matrix(iterable1)
    vector2 = Matrix(iterable2)
    diff = vector1 - vector2
    return norm(diff, p = p)


# Use partial functions to define common norms
euc_norm = partial(norm, p = 2)
max_norm = partial(norm, p = None)
one_norm = partial(norm, p = 1)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = False)