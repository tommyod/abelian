#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from abelian import LCA, LCAFunc, HomLCA, voronoi


def is_close(list1, list2):
    return sum(abs(i-j) for (i,j) in zip(list1, list2)) < 10e-10

class TestLCAFunc:

    def test_natural_transversal(self):
        """
        Test transversal from T^2 to R^2.
        """

        T = LCA([1], [False])
        R = LCA([0], [False])
        A = [[0, 1], [1, -0.5]]
        epimorphism = HomLCA(A, source=R ** 2, target=T ** 2)
        sigma = voronoi(epimorphism, 2)

        assert is_close(sigma([0,0]), [0,0])
        assert is_close(sigma([0.8, 0.8]), [-0.3, -0.2])
        assert is_close(sigma([0.4, 0.8]), [0, 0.4])
        assert is_close(sigma([0.8, 0.4]), [0.3, -0.2])


if __name__ == '__main__':
    tests = TestLCAFunc()
    tests.test_natural_transversal()