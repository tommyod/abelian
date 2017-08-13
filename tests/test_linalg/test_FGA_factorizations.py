#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from sympy import Matrix
from random import randint as ri
from abelian.morphisms import HomFGA
from abelian.linalg.utils import vector_mod_vector

def random_zero_heavy(low, high):
    """
    Draw a random number, with 50% probability of zero.
    """
    return random.choice(list(range(low, high)) + [0]*(high - low))

class TestSNF:

    @classmethod
    def setup_class(cls):
        m, n = ri(3, 5), ri(3, 5)
        A = Matrix(m, n, lambda i, j: random_zero_heavy(-9, 9))
        target = Matrix(m, 1, lambda i, j: random_zero_heavy(5, 50))

        cls.target_vector = target
        cls.phi = HomFGA(A, target = target)



    def test_project_to_source(self):
        """
        Test the periods.
        """

        print(self.phi)
        phi = self.phi.project_to_source()
        periods = Matrix(phi.source.periods)
        zero_vect = periods*0

        #assert vector_mod_vector(phi.A*periods, self.target_vector) ==
        # zero_vect