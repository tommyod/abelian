#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import randint as ri

import pytest
from sympy import Matrix

from abelian.morphisms import HomLCA


class TestHomLCA:

    @pytest.fixture
    def random_homomorphisms(self):
        """ Setup two random homomorphisms phi and psi """
        m, n = 3, 3
        phi = HomLCA(Matrix(m, n, lambda i, j: ri(-5, 5)))
        psi = HomLCA(Matrix(m, n, lambda i, j: ri(-5, 5)))
        return m, n, phi, psi

    def test_horizontal_stacking_property(self, random_homomorphisms):
        """ Test the horizontal stacking property """
        m, n, phi, psi = random_homomorphisms

        # Create random group elements (inputs)
        x = Matrix([ri(-5, 5) for _ in range(n)])
        y = Matrix([ri(-5, 5) for _ in range(n)])

        x_over_y = x.col_join(y)
        stacked = phi.stack_horiz(psi)

        assert stacked(x_over_y) == phi(x) + psi(y)

    def test_vertical_stacking_property(self, random_homomorphisms):
        """ Test the vertical stacking property """
        m, n, phi, psi = random_homomorphisms

        # Create random group elements (inputs)
        x = Matrix([ri(-5, 5) for _ in range(n)])
        stacked = phi.stack_vert(psi)

        assert stacked(x) == phi(x).col_join(psi(x))

    def test_diagonal_stacking_property(self, random_homomorphisms):
        """ Test the diagonal stacking property """
        m, n, phi, psi = random_homomorphisms

        # Create random group elements (inputs)
        x = Matrix([ri(-5, 5) for _ in range(n)])
        y = Matrix([ri(-5, 5) for _ in range(n)])

        x_over_y = x.col_join(y)
        stacked = phi.stack_diag(psi)

        assert stacked(x_over_y) == phi(x).col_join(psi(y))

    def test_call_order(self, random_homomorphisms):
        """ Test that (ab)(x) == a(b(x)) """
        m, n, phi, psi = random_homomorphisms

        # Create random group elements (inputs)
        x = Matrix([ri(-5, 5) for _ in range(n)])
        composed = (phi * psi)

        assert composed(x) == phi(psi(x))
