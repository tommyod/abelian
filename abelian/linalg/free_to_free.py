#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains functions which calculate mapping properties of
free-to-free homomorphisms. All the inputs and outputs are of type
:py:class:`~sympy.matrices.dense.MutableDenseMatrix`.
"""

from sympy import Matrix
from abelian.linalg.factorizations import smith_normal_form
from abelian.linalg.utils import nonzero_diag_as_list


def free_kernel(A):
    """
    Computes the free-to-free kernel of A.

    Let A: Z^n -> Z^m be a free-to-free morphism,
    i.e. a morphism from a free (non-periodic) FGA to
    a free FGA. Associated with the morphism is a kernel
    monomorphism ker(A) such that A * ker(A) * x = 0 for all x.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy integer matrix.

    Returns
    -------
    ker_A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The kernel monomorphism of A.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[1, 0, 1],
    ...             [0, 1, 1]])
    >>> ker_A = free_kernel(A)
    >>> A * ker_A == Matrix([0, 0]) # Check the factorization
    True
    """
    U, S, V = smith_normal_form(A)
    r = len(nonzero_diag_as_list(S))
    return V[:, r:]


def free_cokernel(A):
    """
    Computes the free-to-free cokernel of A.

    Let A: Z^n -> Z^m be a free-to-free morphism,
    i.e. a morphism from a free (non-periodic) FGA to
    a free FGA. Associated with the morphism is a cokernel
    epimorphism coker(A) such that coker(A) * A * x = 0 for all x.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy integer matrix.

    Returns
    -------
    ker_A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The cokernel epimorphism of A.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> from abelian.linalg.utils import matrix_mod_vector
    >>> A = Matrix([[1, 0],
    ...             [0, 1],
    ...             [1, 1]])
    >>> coker_A = free_cokernel(A)
    >>> quotient = free_quotient(A)
    >>> product = matrix_mod_vector(coker_A * A, quotient)
    >>> product == 0 * product
    True
    """
    U, S, V = smith_normal_form(A, compute_transformation=True)
    return U


def free_image(A):
    """
    Computes the free-to-free image of A.

    Let A: Z^n -> Z^m be a free-to-free morphism,
    i.e. a morphism from a free (non-periodic) FGA to
    a free FGA. Associated with the morphism is an image
    monomorphism im(A) such that im(A) * coim(A) = A.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy integer matrix.

    Returns
    -------
    ker_A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The image monomorphism of A.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[1, 0, 1],
    ...             [0, 1, 1]])
    >>> free_image(A) == Matrix.eye(2)
    True
    >>> free_image(A) * free_coimage(A) == A
    True
    """
    U, S, V = smith_normal_form(A)
    r = len(nonzero_diag_as_list(S))
    return U.inv()[:, :r] * S[:r, :r]


def free_coimage(A):
    """
    Computes the free-to-free coimage of A.

    Let A: Z^n -> Z^m be a free-to-free morphism,
    i.e. a morphism from a free (non-periodic) FGA to
    a free FGA. Associated with the morphism is an coimage
    epimorphism coim(A) such that im(A) * coim(A) = A.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy integer matrix.

    Returns
    -------
    ker_A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The coimage epimorphism of A.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[1, 0],
    ...             [0, 1],
    ...             [1, 1]])
    >>> free_coimage(A) == Matrix.eye(2)
    True
    >>> free_image(A) * free_coimage(A) == A
    True
    """
    U, S, V = smith_normal_form(A)
    r = len(nonzero_diag_as_list(S))
    return V.inv()[:r, :]


def free_quotient(A):
    """
    Compute the quotient group Z^m / im(A).

    Let A: Z^n -> Z^m be a free-to-free morphism,
    i.e. a morphism from a free (non-periodic) FGA to
    a free FGA. Associated with the morphism is an cokernel
    epimorphism coker(A), which maps from Z^m to Z^m / im(A).
    TODO READ UP ON THIS AGAIN.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy integer matrix.

    Returns
    -------
    quotient : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The quotient of the morphism A.

    Examples
    ---------
    >>> from sympy import diag
    >>> A = diag(1, 2, 3)
    >>> free_quotient(A) == Matrix([1, 1, 6])
    True
    """
    U, S, V = smith_normal_form(A)
    m, n = A.shape
    diagonal = nonzero_diag_as_list(S)
    r = len(diagonal)
    quotient = diagonal + [0] * (m - r)
    return Matrix(quotient)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = True)


