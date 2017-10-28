#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains stochastic tests kernel, cokernel,
image and coimage algorithms based on the SVD for maps
between vector spaces over R.
"""

from sympy import Matrix
from random import choice, random
from abelian.linalg.factorizations_reals import real_kernel, real_cokernel, \
    real_coimage, real_image

class TestFactorizationsReals:

    @classmethod
    def setup_class(cls):
        """
        Set up a random matrix.
        """
        # Add more zero entries to cover edge cases with higher probability
        numbers = [0] * 10 + list(range(-5, 5)) + [random() for i in range(8)]
        cls.m, cls.n = choice([1,2,3,4]), choice([1,2,3,4])
        cls.A = Matrix(cls.m, cls.n, lambda i, j: choice(numbers))

    def test_kernel(self):
        """
        Test the factorization.
        """

        # Take the kernel
        A = self.A
        m, n = A.shape
        ker = real_kernel(A)

        # Verify norm and dimension
        A_ker_product = A * ker
        small_norm = sum(abs(k) for k in A_ker_product) < 10e-10
        correct_dim = A_ker_product.rows == m

        assert small_norm and correct_dim

    def test_cokernel(self):
        """
        Test the kernel factorization.
        """

        # Take the kernel
        A = self.A
        m, n = A.shape
        coker = real_cokernel(A)

        # Verify norm and dimension
        coker_A_product = coker * A
        small_norm = sum(abs(k) for k in coker_A_product) < 10e-10
        correct_dim = coker_A_product.cols == n

        assert small_norm and correct_dim

    def test_image_coimage(self):
        """
        Test the image/coimage factorization.
        """
        # Take the kernel
        A = self.A
        im = real_image(A)
        coim = real_coimage(A)

        # Verify norm and dimension
        im_coim_product = im * coim
        small_norm = sum(abs(k) for k in (A - im_coim_product)) < 10e-10

        assert small_norm