#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import randint as ri

from sympy import Matrix
import pytest
from abelian import LCA, HomLCA
from tests.utils import random_from_list


@pytest.fixture
def setup():
    """
    Setup a random phi:H -> G:
    """
    m, n = ri(2, 5), ri(2, 5)
    H = LCA(random_from_list(n, [0, 0, 0, 0, 0, 0]))
    G = LCA(random_from_list(m, [0, 0, 0, 5, 8, 9, 10]))
    A = Matrix(m, n, lambda i, j: ri(-5, 5))
    phi = HomLCA(A, source=H, target=G)
    return phi, H, G


def test_homomorphism_property(setup):
    """
    Test the homomorphism property.
    """
    phi, H, G = setup

    x = Matrix([ri(-9, 9) for i in range(len(H))])
    y = Matrix([ri(-9, 9) for i in range(len(H))])

    assert phi(x + y) == G(phi(x) + phi(y))


def test_morphism_from_trivial(setup):
    """
    Test a homomorphism from Z_1.
    """
    phi, H, G = setup

    k = ri(1, 3)
    trivial_source = LCA(random_from_list(k, [1, 1, 1, 1]))

    A = Matrix(len(G), k, lambda i, j: ri(-5, 5))
    phi_triv = HomLCA(A, source=trivial_source, target=G)


def test_identity_morphism(setup):
    """
    Test identity property of a morphism.
    """
    phi, H, G = setup

    Id_G = HomLCA.identity(G)
    Id_H = HomLCA.identity(H)

    assert (phi * Id_H).equal(phi)
    assert (Id_G * phi).equal(phi)


def test_zero_morphism(setup):
    """
    Test identity property of a morphism.
    """
    phi, H, G = setup

    zero = HomLCA.zero(source=H, target=G)
    row_vector = Matrix(1, len(H), lambda i, j: 0)
    first = HomLCA(row_vector, source=H, target=LCA.trivial())

    col_vector = Matrix(len(G), 1, lambda i, j: 0)
    second = HomLCA(col_vector, source=LCA.trivial(), target=G)

    assert (second * first) == zero
