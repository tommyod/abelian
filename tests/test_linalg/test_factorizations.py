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


# Test SNF


@pytest.fixture
def setup_test_SNF():
    """
    Setup random matrices for the Smith normal form.
    """
    m, n = ri(3, 5), ri(3, 5)
    A = Matrix(m, n, lambda i, j: ri(-9, 9))

    U, S, V = smith_normal_form(A)
    return A, U, S, V


def zeros_off_diagonal(S):
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


def positive_diag(S):
    """
    Verifiy that diagonals are positive.
    """
    n = min(S.shape)
    for i in range(n):
        if S[i, i] < 0:
            return False
    return True


def divisibility_diag(S):
    """
    Verify that a_{i}|a_{i+1} for all {i}.
    """
    n = min(S.shape)
    for i in range(n - 1):
        if S[i + 1, i + 1] % S[i, i] != 0:
            return False
    return True


def test_SNF_zeros_off_diagonal(setup_test_SNF):
    _, _, S, _ = setup_test_SNF

    assert zeros_off_diagonal(S)


def test_SNF_positive_diag(setup_test_SNF):
    _, _, S, _ = setup_test_SNF
    assert positive_diag(S)


def test__SNF_divisibility_diag(setup_test_SNF):
    _, _, S, _ = setup_test_SNF
    assert divisibility_diag(S)


def test_SNF_hermite_normal_form(setup_test_SNF):
    A, U, S, V = setup_test_SNF
    assert (U * A * V == S)


def test_SNF_unimodularity(setup_test_SNF):
    _, U, _, V = setup_test_SNF
    assert (V.det() in [1, -1]) and (U.det() in [1, -1])


# Test HNF


@pytest.fixture
def setup_test_HNF():
    """
    Create matrices for testing the Hermite Normal form.
    """
    m, n = ri(3, 5), ri(3, 5)
    A = Matrix(m, n, lambda i, j: ri(-9, 9))

    U, H = hermite_normal_form(A)
    return A, U, H


def positive_pivots(H):
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


def left_smaller_than_pivot(H):
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


def test_HNF_positive_pivots(setup_test_HNF):
    _, _, H = setup_test_HNF
    assert positive_pivots(H)


def test_HNF_smaller_than_pivot(setup_test_HNF):
    _, _, H = setup_test_HNF
    assert left_smaller_than_pivot(H)


def test_hermite_normal_form(setup_test_HNF):
    A, U, H = setup_test_HNF
    assert A * U == H


def test_unimodularity(setup_test_HNF):
    _, U, _ = setup_test_HNF
    assert U.det() in [1, -1]
