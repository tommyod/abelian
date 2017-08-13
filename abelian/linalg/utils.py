#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
from sympy import Matrix, pprint, gcd, lcm
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
        A list of lists with [[col1], [col2], ...].

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
        found_nonzero = False

        # Loop over the rows
        for i in range(0, m):
            if H[i, j] != 0:
                nonzero_cols += 1
                found_nonzero = True
                break
        # TODO: This seems to be correct now. Verify with tests.
        #if not found_nonzero:
            #break
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

    #m, n = S.shape
    #for i in range(0, min(m, n)):
    #    if S[i, i] == 0:
    #        return i
    #else:
    #    return min(m, n)


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


def delete_zero_columns(M):
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
    >>> delete_zero_columns(A) == Matrix([1, 2])
    True
    >>> A = diag(0,1,2)
    >>> A_del = Matrix([[0, 0],
    ...                 [1, 0],
    ...                 [0, 2]])
    >>> delete_zero_columns(A) == A_del
    True
    """

    A = M.copy()
    m, n = A.shape

    cols_to_delete = []
    # Iterate over columns
    for j in range(0, n):
        all_zero = True
        for i in range(0, m):
            if A[i, j] != 0:
                all_zero = False
                break
        # If the column has all zeros, we delete it later
        if all_zero:
            cols_to_delete.append(j)

    # Delete the columns that are to be deleted
    deleted = 0
    for j in cols_to_delete:
        A.col_del(j - deleted)
        deleted += 1

    return A


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
    from sympy import pprint
    for j in cols_to_remove:
        #pprint(new_A)
        #print(j, j - deleted)
        #print('---')
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


    Returns a copy of `A` with every column modded by `mod_col`.

    Parameters
    ----------
    v : :py:class:`~sympy.matrices.dense.MutableDenseMatrix` or a list
        An iterable object with integers. This is the group element.
    mod_vector : :py:class:`~sympy.matrices.dense.MutableDenseMatrix` or a list
        An iterable object with integers. This is the periods of the group.

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

    if len(v) != len(mod_vector):
        raise ValueError('The arguments must have the same length.')

    # Identity element, order 1
        #if all(e == 0 for e in v):
        #   return 1

    # Non identity element in infintite group -> infinite period
        #if all(period == 0 for period in mod_vector):
        #    return 0

    def div(top, bottom):
        try:
            return top // bottom
        except:
            return 1



    gcd_list = [div(order, gcd(element, order)) for
                (element, order) in zip(v, mod_vector)]

    return functools.reduce(lcm, gcd_list)

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = True)

    print(order_of_vector([1, 0, 1], [5, 0, 0]))

