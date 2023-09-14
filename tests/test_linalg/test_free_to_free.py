#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module tests free-to-free kernels, cokernels images and coimages.
It also tests generation of elements by increasing max norm.
All these are stochastics, of dimensions/rank 3-5 typically.
"""

import itertools
from random import choice
from random import randint as ri

import pytest
from sympy import Matrix

from abelian.linalg.free_to_free import free_coimage, free_cokernel, \
    free_image, free_kernel, free_quotient, elements_of_maxnorm_FGA, \
    elements_of_maxnorm
from abelian.linalg.free_to_free import mod
from abelian.linalg.utils import vector_mod_vector


class TestElementsGeneratorFree:
    """
    Test the free elements' generator,
    which generates elements of Z^r with increasing max-norm.
    """

    def test_elements_of_max_norm_num_elements(self):
        """ Verify the number of elements """

        # Random parameter values
        dim = ri(1, 5)
        norm_value = ri(3, 5)

        # Theoretical value
        theoretical_value = (2 * norm_value + 1) ** dim - (2 * norm_value - 1) ** dim

        # Actual value
        generated = list(elements_of_maxnorm(dim, norm_value))
        assert len(set(generated)) == theoretical_value

    def test_elements_of_max_norm_increasing_norm(self):
        """ Verify that the norm is increasing """

        # Random parameter values
        dim = ri(1, 4)
        norm_value = ri(2, 4)

        # Generate elements up to a norm
        generated = []
        for k in range(norm_value):
            generated += list(elements_of_maxnorm(dim, k))

        # Define the maximum norm
        norm = lambda v: max(abs(k) for k in v)

        # Assert that the norm is increasing
        assert all(norm(a) <= norm(b) for a, b in
                   zip(generated[:-1], generated[1:]))


def naive_FGA_elements_by_norm(orders, max_norm_value):
    """
    Naively generated every element in Z_p such
    that max_norm(element) = max_norm_value.

    This algorithm goes through a cartesian product, generates
    everything, projects and checks the norm.
    """

    # Wrap around max norm
    def max_norm_in_finite(a, b):
        return tuple([min(abs(i), abs(i - j)) for i, j in zip(a, b)])

    # Argument into cartesian product and storage for yielded values
    prodarg = [range(-max_norm_value, max_norm_value + 1) for _ in orders]
    yielded = set()

    # Take Cartesian products
    for prod in itertools.product(*prodarg):
        # Project argument
        prod = mod(prod, tuple(orders))
        # If not yielded and correct norm
        if (prod not in yielded) and \
                (max(max_norm_in_finite(prod, tuple(orders))) == max_norm_value):
            yielded.add(prod)
            yield prod


class TestElementsGeneratorFGA:

    def test_free_equality(self):
        """ Verify the number of elements """

        # Random parameter values
        dim = ri(1, 4)
        norm_value = ri(2, 4)

        # Generate with free
        ret1 = list(elements_of_maxnorm(dim, norm_value))
        ret2 = list(elements_of_maxnorm_FGA(dim * [0], norm_value))

        ret1.sort()
        ret2.sort()

        assert ret1 == ret2

    def test_VS_naive_generator(self):
        """
        Test vs the naive generator.
        Make sure the returned values are equal.
        """
        norm_value = ri(0, 4)
        choices = [0] * 5 + [4, 5, 12, 13]
        orders = [choice(choices) for k in range(ri(1, 4))]

        # Return from naive
        ret1 = list(naive_FGA_elements_by_norm(orders, norm_value))

        # Return from algorithm to test against
        ret2 = list(elements_of_maxnorm_FGA(orders, norm_value))

        # Sort and compare
        ret1.sort()
        ret2.sort()
        assert ret1 == ret2


class TestFreeToFree:

    @pytest.fixture
    def setup(self):
        m, n = ri(1, 5), ri(1, 5)
        A = Matrix(m, n, lambda i, j: ri(-9, 9))

        # Compute ker, coker, im, coim and quotient group
        A_ker = free_kernel(A)
        A_coker = free_cokernel(A)
        A_im = free_image(A)
        A_coim = free_coimage(A)
        A_quotient = free_quotient(A)

        return A, A_ker, A_coker, A_im, A_coim, A_quotient

    def test_im_coim(self, setup):
        """ Test the image/coimage factorization of A """
        A, A_ker, A_coker, A_im, A_coim, A_quotient = setup

        assert A_im * A_coim == A

    def test_kernel(self, setup):
        """ Test that the composition of A and ker(A) is zero """
        A, A_ker, A_coker, A_im, A_coim, A_quotient = setup

        sum_entries = sum(abs(entry) for entry in A * A_ker)
        assert sum_entries == 0

    def test_cokernel(self, setup):
        """ Test that the composition of A and ker(A) is zero in the target group """
        A, A_ker, A_coker, A_im, A_coim, A_quotient = setup

        prod = A_coker * A
        m, n = prod.shape
        ones = Matrix(n, 1, list(1 for _ in range(n)))
        prod_in_target_grp = vector_mod_vector(prod * ones, A_quotient)
        assert prod_in_target_grp == A_quotient * 0
