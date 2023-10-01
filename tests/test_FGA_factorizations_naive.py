#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import itertools
import operator
from random import randint as ri

import pytest
from sympy import Matrix, diag

from abelian.linalg.factorizations import smith_normal_form
from abelian.linalg.utils import order_of_vector, remove_zero_columns, \
    nonzero_diag_as_list, vector_mod_vector
from abelian.utils import mod


class TestFGAFactorizationsNaively:

    @pytest.fixture
    def quotient_group(self):
        max_size = 15
        m, n = ri(1, max_size), ri(2, max_size)
        num_generators = ri(1, 2)
        generators = [
            [mod(ri(1, max_size), m), mod(ri(1, max_size), n)] for _ in range(num_generators)
        ]  # yapf. disable
        periods = [order_of_vector(g, [m, n]) for g in generators]

        return generators, periods, m, n

    def test_cokernel_target(self, quotient_group):
        """ Test the quotient group """
        generators, periods, m, n = quotient_group

        # According to proof, this is true
        A = Matrix(generators).T
        ker_pi = remove_zero_columns(diag(m, n))
        U, S, V = smith_normal_form(A.row_join(ker_pi))
        diagonal = nonzero_diag_as_list(S)
        (m, n), r = A.shape, len(diagonal)
        quotient = diagonal + [0] * (m - r)
        print(quotient)

        # Compute the quotient naively. Iterate through all combinations of
        # the generators, store the group elements which we find, and compare
        # this with the order of the group.
        seen_elements = set()
        periods_vector = Matrix([m, n])
        for p in itertools.product(*[range(p) for p in periods]):
            element = vector_mod_vector(A * Matrix(p), periods_vector)
            seen_elements.add(tuple(element))

        naive_quotient = m * n / len(seen_elements)

        assert naive_quotient == 1.
        assert naive_quotient == functools.reduce(operator.mul, quotient)
