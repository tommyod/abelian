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

    @classmethod
    def setup_class(cls):
        """Setup two random homomorphisms phi and psi.
        """
        cls.m, cls.n = 3, 3
        cls.phi = HomLCA(Matrix(cls.m, cls.n, lambda i, j: ri(-5, 5)))
        cls.psi = HomLCA(Matrix(cls.m, cls.n, lambda i, j: ri(-5, 5)))

    def test_stack_horizonally(self):
        """
        Test the horizontal stacking property.
        """
        # Create random group elements (inputs)
        x = Matrix([ri(-5, 5) for i in range(self.n)])
        y = Matrix([ri(-5, 5) for i in range(self.n)])

        x_over_y = x.col_join(y)
        stacked = self.phi.stack_horiz(self.psi)
        assert stacked(x_over_y) == self.phi(x) + self.psi(y)

    def test_stack_vertically(self):
        """
        Test the vertical stacking property.
        """

        # Create random group elements (inputs)
        x = Matrix([ri(-5, 5) for i in range(self.n)])

        stacked = self.phi.stack_vert(self.psi)
        assert stacked(x) == self.phi(x).col_join(self.psi(x))

    def test_stack_diagonally(self):
        """
        Test the diagonal stacking property.
        """

        # Create random group elements (inputs)
        x = Matrix([ri(-5, 5) for i in range(self.n)])
        y = Matrix([ri(-5, 5) for i in range(self.n)])

        x_over_y = x.col_join(y)
        stacked = self.phi.stack_diag(self.psi)

        assert stacked(x_over_y) == self.phi(x).col_join(self.psi(y))

    def test_call_order(self):
        """
        Test that (a \circ b)(x) == a (b(x)).
        """
        # Create random group elements (inputs)
        x = Matrix([ri(-5, 5) for i in range(self.n)])
        composed = (self.phi * self.psi)
        assert composed(x) == self.phi(self.psi(x))