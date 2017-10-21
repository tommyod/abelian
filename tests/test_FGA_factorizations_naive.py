#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from sympy import Matrix, diag
from random import randint as ri
from abelian import HomLCA
from abelian.utils import mod
from abelian.linalg.utils import order_of_vector, remove_zero_columns, \
    nonzero_diag_as_list, vector_mod_vector
from abelian.linalg.factorizations import smith_normal_form
import itertools
from functools import reduce
from operator import mul
import operator
import functools

def random_zero_heavy(low, high):
    """
    Draw a random number, with approx 50% probability of zero.
    """
    return random.choice(list(range(low, high)) + [0]*(high - low))

class TestFGAFactorizationsNaively:

    @classmethod
    def setup_class(cls):
        max_size = 15
        m, n = ri(1, max_size), ri(2, max_size)
        print(m, n)
        num_generators = ri(1, 2)
        generators = [[mod(ri(1, max_size), m), mod(ri(1, max_size), n)]
                      for i in range(num_generators)]
        print(generators)
        periods = [order_of_vector(g, [m, n]) for g in generators]
        print(periods)

        cls.generators = generators
        cls.periods = periods
        cls.m, cls.n = m, n

    def test_cokernel_target(self):
        """
        Test the quotient group.
        """

        # According to proof, this is true
        A = Matrix(self.generators).T
        ker_pi = remove_zero_columns(diag(self.m, self.n))
        U, S, V = smith_normal_form(A.row_join(ker_pi))
        diagonal = nonzero_diag_as_list(S)
        (m, n), r = A.shape, len(diagonal)
        quotient = diagonal + [0] * (m - r)
        print(quotient)

        # Compute the quotient naively. Iterate through all combinations of
        # the generators, store the group elements which we find, and compare
        # this with the order of the group.
        seen_elements = set()
        periods_vector = Matrix([self.m, self.n])
        for p in itertools.product(*[range(p) for p in self.periods]):
            element = vector_mod_vector(A*Matrix(p), periods_vector)
            seen_elements.add(tuple(element))

        naive_quotient = self.m * self.n / len(seen_elements)
        assert naive_quotient % 1 == 0

        assert naive_quotient == functools.reduce(operator.mul, quotient)




if __name__ == '__main__':
    test = TestFGAFactorizationsNaively()
    test.setup_class()
    test.test_cokernel_target()