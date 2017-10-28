#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains functions which calculate mapping properties of
free-to-free homomorphisms. All the inputs and outputs are of type
:py:class:`~sympy.matrices.dense.MutableDenseMatrix`.
"""

import itertools
from sympy import Matrix, Integer, Float, Rational
from abelian.linalg.factorizations import smith_normal_form
from abelian.linalg.utils import nonzero_diag_as_list
from collections.abc import Iterable



def mod(a, b):
    """
    Mod for integers, tuples and lists.

    Parameters
    ----------
    a : int, tuple or list
        The argument.
    b : int, tuple or list
        The order.

    Returns
    -------
    int, tuple or list
        A mod b.

    Examples
    ---------
    >>> mod(7, 5) # Integer data
    2
    >>> mod((5, 8), (4, 4)) # Tuple data
    (1, 0)
    >>> mod([5, 8], [4, 4]) # List data
    [1, 0]
    """
    integer_types = (int, Integer, float, Float, Rational)

    # It's a numeric, non-iterable data type
    if isinstance(a, integer_types) and isinstance(b, integer_types):
        if b == 0:
            return a
        return a % b

    # It's an iterable data type
    if isinstance(a, Iterable) and isinstance(b, Iterable):
        return type(a)([mod(i, j) for i, j in zip(a, b)])

    raise TypeError('Did not recognize data type for modulus.')

def elements_increasing_norm(free_rank, end_value = None):
    """
    Continually yield every element in Z^r of increasing max-norm.

    Parameters
    ----------
    free_rank : int
        The free rank (like dimension) of Z^r, i.e. free_rank = r.

    Yields
    -------
    tuple
        Elements in Z^r with increasing maxnorm.

    Examples
    ---------
    >>> free_rank = 2 # Like dimension
    >>> for count, element in enumerate(elements_increasing_norm(free_rank)):
    ...     if count >= 9:
    ...         break
    ...     print(count, element, max(abs(k) for k in element))
    0 (0, 0) 0
    1 (1, -1) 1
    2 (-1, -1) 1
    3 (1, 0) 1
    4 (-1, 0) 1
    5 (1, 1) 1
    6 (-1, 1) 1
    7 (0, 1) 1
    8 (0, -1) 1
    """
    for maxnorm_value in itertools.count(start = 0, step = 1):
        if end_value is not None:
            if maxnorm_value == end_value:
                break
        yield from elements_of_maxnorm(free_rank, maxnorm_value)

def elements_of_maxnorm_FGA(orders, maxnorm_value):
    """
    Yield every element of Z_`orders` such that max_norm(element) = maxnorm_value.

    Parameters
    ----------
    orders : list
        Orders in Z_orders, where 0 means infinite order,
        i.e. [2, 0] is Z_2 + Z.
    maxnorm_value : int
        The value of the maximum norm of the elements generated.

    Yields
    -------
    tuple
        Elements in Z_orders that satisfy the norm criterion.

    Examples
    ---------
    >>> orders = [0, 0]
    >>> norm_value = 1
    >>> elements = list(elements_of_maxnorm_FGA(orders, norm_value))
    >>> len(elements)
    8
    >>> orders = [0, 3]
    >>> norm_value = 2
    >>> for element in elements_of_maxnorm_FGA(orders, norm_value):
    ...     print(element)
    (2, 2)
    (-2, 2)
    (2, 0)
    (-2, 0)
    (2, 1)
    (-2, 1)
    """

    # The zeroth layer is just (0,0,...), yield this and return to terminate
    if maxnorm_value == 0:
        yield tuple([0] * len(orders))
        return

    # Will be used in the loop, so we compute it out-of-loop here
    dimension = len(orders)

    # The 'wall' is the dimension held constant
    for wallnum, dim in enumerate(orders):

        # If the wall is outside the dimension, skip it
        if (dim != 0) and maxnorm_value > (dim // 2):
            continue

        # Set up the cartesian product argument, making sure to remove
        # boundary elements so they are not yielded twice
        boundary_reduced = [1] * wallnum + [0] * (dimension - wallnum - 1)
        prod_arg = [range(-maxnorm_value + k, maxnorm_value + 1 - k) \
                    for k in boundary_reduced]

        # The dimensions that are not constant
        non_const_dims = orders[:]
        non_const_dims.pop(wallnum)
        # non_const_dims = dimensions[:wallnum] + dimensions[wallnum+1:]

        # Go through every argument in the cartesian product, and
        # reduce the iterator if it's partially outside of the order
        for i in range(len(prod_arg)):
            iterator = prod_arg[i]
            order = non_const_dims[i]

            # If the order is finite, we might be able to reduce the
            # cartesian product by a significant amount
            # The code below does this
            if order != 0:
                old_start, old_stop = iterator.start, iterator.stop
                prod_arg[i] = range(max(-order // 2 + 1, old_start),
                                    min(order // 2 + 1, old_stop))

        # Go through the Cartesian product / hypercube
        for prod in itertools.product(*prod_arg):

            # The first and last part of the element
            first = mod(prod[:wallnum], non_const_dims[:wallnum])
            last = mod(prod[wallnum:], non_const_dims[wallnum:])

            middle1, middle2 = mod(maxnorm_value, dim), mod(-maxnorm_value, dim)
            yield first + (middle1,) + last
            if middle1 != middle2:
                yield first + (middle2,) + last

def elements_of_maxnorm(free_rank, maxnorm_value):
    """
    Yield every element of Z^r such that max_norm(element) = maxnorm_value.

    Parameters
    ----------
    free_rank : int
        The free rank (like dimension) of Z^r, i.e. free_rank = r.
    maxnorm_value : int
        The value of the maximum norm of the elements generated.

    Yields
    -------
    tuple
        Elements in Z^r that satisfy the norm criterion.

    Examples
    ---------
    >>> free_rank = 3 # Like dimension
    >>> maxnorm_value = 4
    >>> elements = list(elements_of_maxnorm(free_rank, maxnorm_value))
    >>> # Verify that the max norm is the correct value
    >>> all(max(abs(k) for k in e) for e in elements)
    True
    >>> # Verify the number of elements
    >>> n = maxnorm_value
    >>> len(elements) == ((2*n + 1)**free_rank - (2*n - 1)**free_rank)
    True
    """
    if maxnorm_value == 0:
        yield tuple([0] * free_rank)
        return

    # There are two 'walls' per dimension, front and back
    for wall in range(free_rank):

        # In each wall, the boundaries must shrink, two at a time
        boundary_reduced = [1] * wall + [0] * (free_rank - wall - 1)

        # The arguments into the cartesian product
        prod_arg = [range(-maxnorm_value + k, maxnorm_value + 1 - k)\
                    for k in boundary_reduced]

        # Take cartesian products along the boundaries of the r-dimensional
        # cube. Yield from opposite sides of the hypercube
        for boundary_element in itertools.product(*prod_arg):
            start, end = boundary_element[:wall], boundary_element[wall:]
            yield start + (maxnorm_value,) + end
            yield start + (-maxnorm_value,) + end


def free_kernel(A):
    """
    Computes the free-to-free kernel monomorphism of A.

    Let :math:`A: \mathbb{Z}^n -> \mathbb{Z}^m` be a homomorphism from
    a free (infinite order) finitely generated Abelian group (FGA) to another
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
    a free (infinite order) finitely generated Abelian group (FGA) to another
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
    a free (infinite order) finitely generated Abelian group (FGA) to another
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
    a free (infinite order) finitely generated Abelian group (FGA) to another
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
    a free (infinite order) finitely generated Abelian group (FGA) to another
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