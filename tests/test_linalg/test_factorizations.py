#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import randint as ri
from sympy import Matrix
from abelian.linalg.factorizations import hermite_normal_form, \
    smith_normal_form


class TestSNF:

    @classmethod
    def setup_class(cls):
        m, n = ri(3, 5), ri(3, 5)
        A = Matrix(m, n, lambda i, j: ri(-9, 9))

        U, S, V = smith_normal_form(A)
        cls.A = A
        cls.U = U
        cls.S = S
        cls.V = V

    def zeros_off_diagonal(self, S):
        """
        Verify that off-diagonals are zero.
        """
        m, n = S.shape
        for i in range(m):
            for j in range(n):
                if i == j:
                    continue
                if S[i, j] != 0:
                    return False
        return True

    def positive_diag(self, S):
        """
        Verifiy that diagonals are positive.
        """
        n = min(S.shape)
        for i in range(n):
            if S[i, i] < 0:
                return False
        return True

    def divisibility_diag(self, S):
        """
        Verify that a_{i}|a_{i+1} for all {i}.
        """
        n = min(S.shape)
        for i in range(n - 1):
            if S[i + 1, i + 1] % S[i, i] != 0:
                return False
        return True

    def test_zeros_off_diagonal(self):
        assert (self.zeros_off_diagonal(self.S))

    def test_positive_diag(self):
        assert (self.positive_diag(self.S))

    def test_divisibility_diag(self):
        assert (self.divisibility_diag(self.S))

    def test_hermite_normal_form(self):
        assert (self.U * self.A * self.V == self.S)

    def test_unimodularity(self):
        assert (self.V.det() in [1, -1]) and (self.U.det() in [1, -1])


class TestHNF:
    @classmethod
    def setup_class(cls):
        """
        Setup state specific to execution of class, which contains tests.
        """
        m, n = ri(3, 5), ri(3, 5)
        A = Matrix(m, n, lambda i, j: ri(-9, 9))

        U, H = hermite_normal_form(A)
        cls.A = A
        cls.U = U
        cls.H = H

    def positive_pivots(self, H):
        """
        Check that all pivots are positive.
        """
        m, n = H.shape
        for j in range(n):
            for i in range(m):
                if H[i, j] == 0:
                    continue
                if H[i, j] > 0:
                    break
                if H[i, j] < 0:
                    return False

        return True

    def left_smaller_than_pivot(self, H):
        """
        Check that all elements k to the left of a pivot h are in the range 0 <= k < h.
        """
        m, n = H.shape
        for j in range(n):
            for i in range(m):
                if H[i, j] > 0:
                    if not all(0 <= H[i, k] < H[i, j] for k in range(0, j)):
                        return False
                    break

        return True

    def test_positive_pivots(self):
        assert self.positive_pivots(self.H)

    def test_smaller_than_pivot(self):
        assert self.left_smaller_than_pivot(self.H)

    def test_hermite_normal_form(self):
        assert self.A * self.U == self.H

    def test_unimodularity(self):
        assert self.U.det() in [1, -1]
