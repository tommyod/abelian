#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module tests free-to-free kernels, cokernels images and coimages.
It also test generation of elements by increasing max norm.
All tese are stochastics, of dimensions/rank 3-5 typically.
"""

import itertools
from random import randint as ri
from random import choice
from sympy import Matrix

from abelian.linalg.free_to_free import free_coimage, free_cokernel, \
    free_image, free_kernel, free_quotient, elements_of_maxnorm_FGA, \
    elements_of_maxnorm
from abelian.linalg.utils import vector_mod_vector
from abelian.linalg.free_to_free import mod

class TestElementsGeneratorFree:
    """
    Test the free elements generator,
    which generates elements of Z^r with increasing max-norm.
    """

    def test_elements_of_maxnorm_num_elements(self):
        """
        Verify the number of elements.
        """

        # Random parameter values
        dim = ri(1, 5)
        normvalue = ri(3, 5)

        # Theoretical value
        theoretical_value = (2*normvalue + 1)**dim - (2*normvalue - 1)**dim

        # Actual value
        generated = list(elements_of_maxnorm(dim, normvalue))
        assert len(set(generated)) == theoretical_value


    def test_elements_of_maxnorm_increasing_norm(self):
        """
        Verify that the norm is increasing.
        """

        # Random parameter values
        dim = ri(1, 4)
        normvalue = ri(2, 4)

        # Generate elements up to a norm
        generated = []
        for k in range(normvalue):
            generated += list(elements_of_maxnorm(dim, k))

        # Define the maximum norm
        norm = lambda v : max(abs(k) for k in v)

        # Assert that the norm is increasing
        assert all(norm(a)<= norm(b) for a, b in
                   zip(generated[:-1], generated[1:]))


def naive_FGA_elements_by_norm(orders, maxnorm_value):
    """
    Naively generated every element in Z_p such
    that maxnorm(element) = maxnorm_value.

    This algorithm goes through a cartesian product, generates
    everything, projects and checks the norm.
    """

    # Wrap around max norm
    def maxnorm_in_finite(a, b):
        return tuple([min(abs(i), abs(i - j)) for i, j in zip(a, b)])

    # Argument into cartesian product and storage for yielded values
    prodarg = [range(-maxnorm_value, maxnorm_value + 1) for o in orders]
    yielded = set()

    # Take Cartesian products
    for prod in itertools.product(*prodarg):
        # Project argument
        prod = mod(prod, tuple(orders))
        # If not yielded and correct norm
        if (prod not in yielded) and \
                (max(maxnorm_in_finite(prod, tuple(orders))) == maxnorm_value):
            yielded.add(prod)
            yield prod

class TestElementsGeneratorFGA:

    def test_free_equality(self):
        """
        Verify the number of elements.
        """

        # Random parameter values
        dim = ri(1, 4)
        normvalue = ri(2, 4)

        # Generate with free
        ret1 = list(elements_of_maxnorm(dim, normvalue))
        ret2 = list(elements_of_maxnorm_FGA(dim*[0], normvalue))

        ret1.sort()
        ret2.sort()

        assert ret1 == ret2

    def test_VS_naive_generator(self):
        """
        Test vs the naive generator.
        Make sure the returned values are equal.
        """
        normvalue = ri(0, 4)
        choices = [0]*5 + [4,5,12,13]
        orders = [choice(choices) for k in range(ri(1,4))]

        # Return from naive
        ret1 = list(naive_FGA_elements_by_norm(orders, normvalue))

        # Return from algorithm to test against
        ret2 = list(elements_of_maxnorm_FGA(orders, normvalue))

        # Sort and compare
        ret1.sort()
        ret2.sort()
        assert ret1 == ret2






class TestFreeToFree:
    @classmethod
    def setup_class(cls):
        m, n = ri(3, 5), ri(3, 5)
        A = Matrix(m, n, lambda i, j: ri(-9, 9))
        cls.A = A

        # Compute ker, coker, im, coim and quotient group
        cls.A_ker = free_kernel(A)
        cls.A_coker = free_cokernel(A)
        cls.A_im = free_image(A)
        cls.A_coim = free_coimage(A)
        cls.A_quotient = free_quotient(A)

    def test_im_coim(self):
        """
        Test the image/coimage factorization of A.
        """
        assert (self.A_im * self.A_coim == self.A)

    def test_kernel(self):
        """
        Test that the composition of A and ker(A) is zero.
        """
        sum_entries = sum(abs(entry) for entry in self.A * self.A_ker)
        assert (sum_entries == 0)

    def test_cokernel(self):
        """
        Test that the composition of A and ker(A) is zero in the target group.
        """
        prod = self.A_coker * self.A
        m, n = prod.shape
        ones = Matrix(n, 1, list(1 for i in range(n)))
        prod_in_target_grp = vector_mod_vector(prod * ones, self.A_quotient)
        assert (prod_in_target_grp == self.A_quotient * 0)
