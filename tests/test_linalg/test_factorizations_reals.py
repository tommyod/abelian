#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains stochastic tests' kernel, cokernel,
image and coimage algorithms based on the SVD for maps
between vector spaces over R.
"""

from random import choice, random

import pytest
from sympy import Matrix

from abelian.linalg.factorizations_reals import real_kernel, real_cokernel, \
    real_coimage, real_image


class TestFactorizationsReals:

    @pytest.fixture
    def random_matrix(self):
        """ Set up a random matrix """
        # Add more zero entries to cover edge cases with higher probability
        numbers = [0] * 10 + list(range(-5, 5)) + [random() for _ in range(8)]
        m, n = choice([1, 2, 3, 4, 5]), choice([1, 2, 3, 4, 5])
        A = Matrix(m, n, lambda i, j: choice(numbers))
        return A, m, n

    def test_kernel(self, random_matrix):
        """ Test the factorization """
        A, m, n = random_matrix

        # Take the kernel
        ker = real_kernel(A)

        # Verify norm and dimension
        A_ker_product = A * ker
        small_norm = sum(abs(k) for k in A_ker_product) < 10e-10
        correct_dim = A_ker_product.rows == m

        assert small_norm and correct_dim

    def test_cokernel(self, random_matrix):
        """ Test the kernel factorization """
        A, m, n = random_matrix

        # Take the cokernel
        coker = real_cokernel(A)

        # Verify norm and dimension
        coker_A_product = coker * A
        small_norm = sum(abs(k) for k in coker_A_product) < 10e-10
        correct_dim = coker_A_product.cols == n

        assert small_norm and correct_dim

    def test_image_coimage(self, random_matrix):
        """ Test the image/coimage factorization """
        A, m, n = random_matrix

        # Take the kernel
        im = real_image(A)
        coim = real_coimage(A)

        # Verify norm and dimension
        im_coim_product = im * coim
        assert sum(abs(k) for k in (A - im_coim_product)) < 10e-10
