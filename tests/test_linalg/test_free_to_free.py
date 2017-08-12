#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import randint as ri
from sympy import Matrix

from abelian.linalg.free_to_free import free_coimage, free_cokernel, \
    free_image, free_kernel, free_quotient
from abelian.linalg.utils import vector_mod_vector


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
