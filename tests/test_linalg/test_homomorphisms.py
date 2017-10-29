#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from sympy import Matrix
from random import randint as ri
from abelian.morphisms import HomLCA

def random_zero_heavy(low, high):
    """
    Draw a random number, with approx 50% probability of zero.
    """
    return random.choice(list(range(low, high)) + [0]*(high - low))

class TestHomLCA:

    @staticmethod
    def setup():
        """Setup two random homomorphisms phi and psi.
        """
        m, n = 3, 3
        phi = HomLCA(Matrix(m, n, lambda i, j: ri(-5, 5)))
        psi = HomLCA(Matrix(m, n, lambda i, j: ri(-5, 5)))
        return m, n, phi, psi

    def test_stack_horizonally(self):
        """
        Test the horizontal stacking property.
        """
        m, n, phi, psi = self.setup()

        # Create random group elements (inputs)
        x = Matrix([ri(-5, 5) for i in range(n)])
        y = Matrix([ri(-5, 5) for i in range(n)])

        x_over_y = x.col_join(y)
        stacked = phi.stack_horiz(psi)
        assert stacked(x_over_y) == phi(x) + psi(y)

    def test_stack_vertically(self):
        """
        Test the vertical stacking property.
        """
        m, n, phi, psi = self.setup()


        # Create random group elements (inputs)
        x = Matrix([ri(-5, 5) for i in range(n)])

        stacked = phi.stack_vert(psi)
        assert stacked(x) == phi(x).col_join(psi(x))

    def test_stack_diagonally(self):
        """
        Test the diagonal stacking property.
        """
        m, n, phi, psi = self.setup()


        # Create random group elements (inputs)
        x = Matrix([ri(-5, 5) for i in range(n)])
        y = Matrix([ri(-5, 5) for i in range(n)])

        x_over_y = x.col_join(y)
        stacked = phi.stack_diag(psi)

        assert stacked(x_over_y) == phi(x).col_join(psi(y))

    def test_call_order(self):
        """
        Test that (a \circ b)(x) == a (b(x)).
        """
        m, n, phi, psi = self.setup()

        # Create random group elements (inputs)
        x = Matrix([ri(-5, 5) for i in range(n)])
        composed = (phi * psi)
        assert composed(x) == phi(psi(x))