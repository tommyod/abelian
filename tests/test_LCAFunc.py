#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import isclose

import pytest

from abelian import LCA, HomLCA, voronoi


class TestLCAFunc:

    @pytest.fixture
    def sigma(self):
        T = LCA([1], [False])
        R = LCA([0], [False])
        A = [[0, 1], [1, -0.5]]

        epimorphism = HomLCA(A, source=R ** 2, target=T ** 2)
        sigma = voronoi(epimorphism, 2)

        return sigma

    @pytest.mark.parametrize(
        'input_list,  output_list',
        [
            ([0, 0], [0, 0]),
            ([0.8, 0.8], [-0.3, -0.2]),
            ([0.4, 0.8], [0, 0.4]),
            ([0.8, 0.4], [0.3, -0.2])
        ]
    )
    def test_natural_transversal_real(self, input_list, output_list, sigma):
        """ Test transversal from T^2 to R^2 """

        for input_value, excepted_value in zip(sigma(input_list), output_list):
            assert isclose(input_value, excepted_value, abs_tol=10e-10)
