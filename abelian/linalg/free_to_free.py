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
    Computes the free-to-free kernel monomorphism of A.

    Let :math:`A: \mathbb{Z}^n -> \mathbb{Z}^m` be a homomorphism from
    a free (non-periodic) finitely generated Abelian group (FGA) to another
    free FGA. Associated with this homomorphism is the kernel monomorphism.
    The kernel monomorphism has the property that
    :math:`A \circ \operatorname{ker}(A) = \mathbf{0}`, where :math:`\mathbf{0}`
    denotes the zero morphism.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy integer matrix.

    Returns
    -------
    ker_A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The kernel monomorphism associated with `A`.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[1, 0, 1],
    ...             [0, 1, 1]])
    >>> ker_A = free_kernel(A)
    >>> # Verify the factorization
    >>> A * ker_A == Matrix([0, 0])
    True
    """
    U, S, V = smith_normal_form(A)
    r = len(nonzero_diag_as_list(S))
    return V[:, r:]


def free_cokernel(A):
    """
    Computes the free-to-free cokernel epimorphism of A.

    Let :math:`A: \mathbb{Z}^n -> \mathbb{Z}^m` be a homomorphism from
    a free (non-periodic) finitely generated Abelian group (FGA) to another
    free FGA. Associated with this homomorphism is the cokernel epimorphism.
    The cokernel epimorphism has the property that
    :math:`\operatorname{coker}(A) \circ A = \mathbf{0}`, where
    :math:`\mathbf{0}` denotes the zero morphism.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy integer matrix.

    Returns
    -------
    coker_A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The cokernel epimorphism associated with `A`.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> from abelian.linalg.utils import matrix_mod_vector
    >>> A = Matrix([[1, 0],
    ...             [0, 1],
    ...             [1, 1]])
    >>> coker_A = free_cokernel(A)
    >>> quotient = free_quotient(A)
    >>> # Compute coker(A) * A and verify that it's 0 in the
    >>> # target group of coker(A).
    >>> product = matrix_mod_vector(coker_A * A, quotient)
    >>> product == 0 * product
    True
    """
    U, S, V = smith_normal_form(A, compute_unimod=True)
    return U


def free_image(A):
    """
    Computes the free-to-free image monomorphism of A.

    Let :math:`A: \mathbb{Z}^n -> \mathbb{Z}^m` be a homomorphism from
    a free (non-periodic) finitely generated Abelian group (FGA) to another
    free FGA. Associated with this homomorphism is the image monomorphism.
    The image monomorphism has the property that :math:`A` factors through
    the composition of the coimage and image morphisms, i.e.
    :math:`\operatorname{im}(A) \circ \operatorname{coim}(A) = A`.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy integer matrix.

    Returns
    -------
    im_A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The image monomorphism associated with `A`.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[1, 0, 1],
    ...             [0, 1, 1]])
    >>> # Clearly the image is the identity matrix
    >>> free_image(A) == Matrix.eye(2)
    True
    >>> # Verify the image(A) * coimage(A) = A factorization
    >>> free_image(A) * free_coimage(A) == A
    True
    """
    U, S, V = smith_normal_form(A)
    r = len(nonzero_diag_as_list(S))
    return U.inv()[:, :r] * S[:r, :r]


def free_coimage(A):
    """
    Computes the free-to-free coimage epimorphism of A.

    Let :math:`A: \mathbb{Z}^n -> \mathbb{Z}^m` be a homomorphism from
    a free (non-periodic) finitely generated Abelian group (FGA) to another
    free FGA. Associated with this homomorphism is the coimage epimorphism.
    The coimage epimorphism has the property that :math:`A` factors through
    the composition of the coimage and image morphisms, i.e.
    :math:`\operatorname{im}(A) \circ \operatorname{coim}(A) = A`.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy integer matrix.

    Returns
    -------
    coim_A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The coimage epimorphism associated with A.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[1, 0],
    ...             [0, 1],
    ...             [1, 1]])
    >>> # Clearly the image is A itself, so coim(A) must be I
    >>> free_coimage(A) == Matrix.eye(2)
    True
    >>> # Verify the image(A) * coimage(A) = A factorization
    >>> free_image(A) * free_coimage(A) == A
    True
    """
    U, S, V = smith_normal_form(A)
    r = len(nonzero_diag_as_list(S))
    return V.inv()[:r, :]


def free_quotient(A):
    """
    Compute the quotient group Z^m / im(A).

    Let :math:`A: \mathbb{Z}^n -> \mathbb{Z}^m` be a homomorphism from
    a free (non-periodic) finitely generated Abelian group (FGA) to another
    free FGA. Associated with this homomorphism is the cokernel epimorphism,
    which maps from :math:`A: \mathbb{Z}^n` to
    :math:`A: \mathbb{Z}^m / \operatorname{im}(A)`.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A sympy integer matrix.

    Returns
    -------
    quotient : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The structure of the quotient group target(A)/im(A).

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
    doctest.testmod(verbose = False)