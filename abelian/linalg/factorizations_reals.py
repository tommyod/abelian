#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains functions which calculate mapping properties of
homomorphisms between R^n and R^m using the singular value decomposition (SVD).
All the inputs and outputs are of type
:py:class:`~sympy.matrices.dense.MutableDenseMatrix`.
"""

from sympy import Matrix, diag
from abelian.linalg.utils import columns_as_list
import numpy as np

def numerical_SVD(A):
    """
    Compute U,S,V such that U*S*V = A.

    The input is converted to numerical data, the SVD is computed using
    the np.linalg.svd routine, which wraps the LAPACK routine _gesdd.
    The data is then converted to a sympy matrix and returned.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy matrix.

    Returns
    -------
    U : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A (close to) orthogonal sympy matrix.

    S : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A diagonal sympy matrix matrix.

    V : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A (close to) orthogonal sympy matrix.

    Examples
    ---------
    >>> A = Matrix([[1, 2], [3, 4]])
    >>> U, S, V = numerical_SVD(A)
    >>> # U is orthogonal (up to machine precision or so)
    >>> abs(abs(U.det()) - 1) < 10e-10
    True
    >>> # Verify that the decomposition is close to the original
    >>> sum(abs(k) for k in (U*S*V - A))  < 10e-10
    True
    """
    # Convert A to a numerical type instead of symbolic
    A_numeric = np.array(columns_as_list(A), dtype=float).T

    # Take the singular value decomposition
    U, S, V = np.linalg.svd(A_numeric, full_matrices=True)

    # Convert to sympy data types
    U, S, V = Matrix(U.tolist()), diag(*S.tolist()), Matrix(V.tolist())
    return U, S, V

def numerical_rank(A):
    """
    Convert to numerical matrix and compute rank.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy matrix.

    Returns
    -------
    r : int
        The rank of A.

    Examples
    ---------
    >>> A = Matrix([[1, 2], [3, 4]])
    >>> numerical_rank(A)
    2
    >>> A = Matrix([[0, 0], [0, 10e-10]])
    >>> numerical_rank(A)
    1
    """
    # Convert A to a numerical type instead of symbolic
    A_numeric = np.array(columns_as_list(A), dtype=float).T

    # Compute the rank and return
    return np.linalg.matrix_rank(A_numeric)

def real_kernel(A):
    """
    Find the kernel of A, when the entries are real.

    Converts the matrix to a numerical input, computes the SVD,
    finds the kernel monomorphism (null space of A), converts back
    to a sympy-matrix and returns.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy matrix.

    Returns
    -------
    K : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The kernel of A.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[1, 0, 1],
    ...             [0, 1, 1],
    ...             [2, 2, 4]])
    >>> ker = real_kernel(A)
    >>> # Verify the decomposition
    >>> sum(abs(k) for k in (A * ker)) < 10e-15
    True
    """

    # Get size, rank and decompose A
    m, n = A.shape
    r = numerical_rank(A)
    U, S, V = numerical_SVD(A)

    # Take transpose (inverse) of V
    V = V.T

    # Return the last (n-r) columns of V
    if (n-r) != 0:
        return V[:, -(n-r):]
    else:
        # Return an empty matrix along one dimension
        return V[:, :0]

def real_cokernel(A):
    """
    Find the cokernel of A, when the entries are real.

    Converts the matrix to a numerical input, computes the SVD,
    finds the cokernel epimorphism (null space of A^T), converts back
    to a sympy-matrix and returns.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy matrix.

    Returns
    -------
    K : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The cokernel of A.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[1, 0],
    ...             [0, 1],
    ...             [2, 2]])
    >>> coker = real_cokernel(A)
    >>> # Verify the decomposition
    >>> sum(abs(k) for k in (coker * A)) < 10e-15
    True
    """

    # Get size, rank and decompose A
    m, n = A.shape
    r = numerical_rank(A)
    U, S, V = numerical_SVD(A)

    # Take the transpose/inverse
    U = U.T

    # Return the last (m-r) rows of A
    if (m - r) != 0:
        return U[-(m - r):, :]
    else:
        return U[:0, :]

def real_image(A):
    """
    Find the image of A, when the entries are real.

    Converts the matrix to a numerical input, computes the SVD,
    finds the image monomorphism (column space), converts back
    to a sympy-matrix and returns.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy matrix.

    Returns
    -------
    K : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The image of A.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[1, 0],
    ...             [0, 1],
    ...             [1, 1]])
    >>> im = real_image(A)
    >>> coim = real_coimage(A)
    >>> # Verify the decomposition
    >>> sum(abs(k) for k in (A - im * coim)) < 10e-15
    True
    """

    # Get the rank and decompose A, return first r columns of U
    r = numerical_rank(A)
    U, S, V = numerical_SVD(A)
    return U[:, :r] * S[:r, :r]


def real_coimage(A):
    """
    Find the coimage of A, when the entries are real.

    Converts the matrix to a numerical input, computes the SVD,
    finds the coimage epimorphism (row space of A), converts back
    to a sympy-matrix and returns.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy matrix.

    Returns
    -------
    K : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The coimage of A.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[1, 0, 0],
    ...             [0, 1, 0]])
    >>> im = real_image(A)
    >>> coim = real_coimage(A)
    >>> # Verify the decomposition
    >>> sum(abs(k) for k in (A - im * coim)) < 10e-15
    True
    """

    # Get rank, decompose A and return the first r rows of V
    r = numerical_rank(A)
    U, S, V = numerical_SVD(A)
    return V[:r, :]


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = False)

