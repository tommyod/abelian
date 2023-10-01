#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains stochastic tests for the SNF and HNF.
The tests are based on the mathematical definitions of the
decompositions, and run on random matrices of size 3-5.
"""

from random import randint as ri

import pytest
from sympy import Matrix

from abelian.linalg.factorizations import hermite_normal_form, \
    smith_normal_form


class TestSNF:

    @pytest.fixture
    def smith_normal_form_from_random_matrix(self):
        """ Setup random matrices for the Smith normal form """
        m, n = ri(3, 5), ri(3, 5)
        A = Matrix(m, n, lambda i, j: ri(-9, 9))

        U, S, V = smith_normal_form(A)
        return A, U, S, V

    @staticmethod
    def zeros_off_diagonal(S):
        """ Verify that off-diagonals are zero """
        m, n = S.shape
        for i in range(m):
            for j in range(n):
                if i == j:
                    continue
                if S[i, j] != 0:
                    return False
        return True

    @staticmethod
    def positive_diag(S):
        """ Verify that diagonals are positive """
        n = min(S.shape)
        for i in range(n):
            if S[i, i] < 0:
                return False
        return True

    @staticmethod
    def divisibility_diag(S):
        """ Verify that a_{i}|a_{i+1} for all {i} """
        n = min(S.shape)
        for i in range(n - 1):
            if S[i + 1, i + 1] % S[i, i] != 0:
                return False
        return True

    def test_zeros_off_diagonal(self, smith_normal_form_from_random_matrix):
        _, _, S, _ = smith_normal_form_from_random_matrix
        assert self.zeros_off_diagonal(S)

    def test_positive_diag(self, smith_normal_form_from_random_matrix):
        _, _, S, _ = smith_normal_form_from_random_matrix
        assert self.positive_diag(S)

    def test_divisibility_diag(self, smith_normal_form_from_random_matrix):
        _, _, S, _ = smith_normal_form_from_random_matrix
        assert self.divisibility_diag(S)

    def test_hermite_normal_form(self, smith_normal_form_from_random_matrix):
        A, U, S, V = smith_normal_form_from_random_matrix
        assert U * A * V == S

    def test_unimodularity(self, smith_normal_form_from_random_matrix):
        A, U, S, V = smith_normal_form_from_random_matrix
        assert V.det() in [1, -1] and U.det() in [1, -1]


class TestHNF:

    @pytest.fixture
    def hermitian_normal_form_from_random_matrix(self):
        """ Create matrices for testing the Hermite Normal form"""
        m, n = ri(3, 5), ri(3, 5)
        A = Matrix(m, n, lambda i, j: ri(-9, 9))
        U, H = hermite_normal_form(A)

        return A, U, H

    @staticmethod
    def positive_pivots(H):
        """ Check that all pivots are positive """
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

    @staticmethod
    def left_smaller_than_pivot(H):
        """ Check that all elements k to the left of a pivot h are in the range 0 <= k < h """
        m, n = H.shape
        for j in range(n):
            for i in range(m):
                if H[i, j] > 0:
                    if not all(0 <= H[i, k] < H[i, j] for k in range(0, j)):
                        return False
                    break
        return True

    def test_positive_pivots(self, hermitian_normal_form_from_random_matrix):
        _, _, H = hermitian_normal_form_from_random_matrix
        assert self.positive_pivots(H)

    def test_smaller_than_pivot(self, hermitian_normal_form_from_random_matrix):
        _, _, H = hermitian_normal_form_from_random_matrix
        assert self.left_smaller_than_pivot(H)

    def test_hermite_normal_form(self, hermitian_normal_form_from_random_matrix):
        A, U, H = hermitian_normal_form_from_random_matrix
        assert A * U == H

    def test_unimodularity(self, hermitian_normal_form_from_random_matrix):
        _, U, _ = hermitian_normal_form_from_random_matrix
        assert U.det() in [1, -1]
