#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from random import randint as ri
from abelian import LCA, HomLCA
from sympy import Matrix

def random_from_list(number, list_to_take_from):
    """
    Draw several random values from the same list.
    """
    return [random.choice(list_to_take_from) for i in range(number)]

class TestHomLCA:

    @classmethod
    def setup_class(cls):
        """
        Setup a random phi:H -> G:
        """
        m, n = ri(2, 5), ri(2, 5)
        cls.H = LCA(random_from_list(n, [0,0,0,0,0,0]))
        cls.G = LCA(random_from_list(m, [0, 0, 0, 5, 8, 9, 10]))
        A = Matrix(m, n, lambda i,j : ri(-5,5))
        cls.phi = HomLCA(A, source = cls.H, target = cls.G)

    def test_homomorphism_property(self):
        """
        Test the homomorphism property.
        """
        H = self.H
        G = self.G
        phi = self.phi

        x = Matrix([ri(-9, 9) for i in range(len(H))])
        y = Matrix([ri(-9, 9) for i in range(len(H))])

        assert phi(x + y) == G(phi(x) + phi(y))

    def test_morphism_from_trivial(self):
        """
        Test a homomorphism from Z_1.
        """
        k = ri(1, 3)
        trivial_source = LCA(random_from_list(k, [1, 1, 1, 1]))
        G = self.G

        A = Matrix(len(G), k, lambda i,j : ri(-5, 5))
        phi_triv = HomLCA(A, source = trivial_source, target = G)

    def test_identity_morphism(self):
        """
        Test identity property of a morphism.
        """
        H = self.H
        G = self.G
        phi = self.phi

        Id_G = HomLCA.identity(G)
        Id_H = HomLCA.identity(H)

        assert (phi * Id_H).equal(phi)
        assert (Id_G * phi).equal(phi)

    def test_zero_morphism(self):
        """
        Test identity property of a morphism.
        """
        H = self.H
        G = self.G
        zero = HomLCA.zero(source = H, target = G)

        H_len = len(H)
        row_vector = Matrix(1, len(H), lambda i, j: 0)
        first = HomLCA(row_vector, source = H, target = LCA.trivial())

        col_vector = Matrix(len(G), 1, lambda i, j: 0)
        second = HomLCA(col_vector, source=LCA.trivial(), target=G)

        assert (second * first) == zero



if __name__ == '__main__':
    t = TestHomLCA()
    t.setup_class()
    t.test_homomorphism_property()