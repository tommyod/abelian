#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import randint as ri

import pytest
from sympy import Matrix

from abelian import LCA, HomLCA
from tests.utils import random_from_list


class TestHomLCA:

    @pytest.fixture
    def random_phi(self):
        """ Set up a random phi:H -> G """
        m, n = ri(2, 5), ri(2, 5)
        H = LCA(random_from_list(n, [0, 0, 0, 0, 0, 0]))
        G = LCA(random_from_list(m, [0, 0, 0, 5, 8, 9, 10]))
        A = Matrix(m, n, lambda i, j: ri(-5, 5))
        phi = HomLCA(A, source=H, target=G)
        return phi, H, G

    def test_homomorphism_property(self, random_phi):
        """ Test the homomorphism property """
        phi, H, G = random_phi

        x = Matrix([ri(-9, 9) for _ in range(len(H))])
        y = Matrix([ri(-9, 9) for _ in range(len(H))])

        assert phi(x + y) == G(phi(x) + phi(y))

    # FIXME: this test is doing nothing
    def test_morphism_from_trivial(self, random_phi):
        """ Test a homomorphism from Z_1 """
        phi, H, G = random_phi

        k = ri(1, 3)
        trivial_source = LCA(random_from_list(k, [1, 1, 1, 1]))

        A = Matrix(len(G), k, lambda i, j: ri(-5, 5))
        phi_triv = HomLCA(A, source=trivial_source, target=G)

    def test_identity_morphism(self, random_phi):
        """ Test identity property of a morphism """
        phi, H, G = random_phi

        id_G = HomLCA.identity(G)
        id_H = HomLCA.identity(H)

        assert (phi * id_H).equal(phi) and (id_G * phi).equal(phi)

    def test_zero_morphism(self, random_phi):
        """ Test identity property of a morphism """
        phi, H, G = random_phi

        zero = HomLCA.zero(source=H, target=G)
        row_vector = Matrix(1, len(H), lambda i, j: 0)
        first = HomLCA(row_vector, source=H, target=LCA.trivial())

        col_vector = Matrix(len(G), 1, lambda i, j: 0)
        second = HomLCA(col_vector, source=LCA.trivial(), target=G)

        assert second * first == zero
