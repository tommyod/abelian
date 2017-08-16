#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from sympy import Matrix
from random import randint as ri
from abelian.morphisms import HomFGA

def random_zero_heavy(low, high):
    """
    Draw a random number, with approx 50% probability of zero.
    """
    return random.choice(list(range(low, high)) + [0]*(high - low))

class TestSNF:

    @classmethod
    def setup_class(cls):
        m, n = ri(2, 4), ri(2, 4)
        A = Matrix(m, n, lambda i, j: random_zero_heavy(-9, 9))
        target = Matrix(m, 1, lambda i, j: random_zero_heavy(5, 50))

        cls.target_vector = target
        cls.phi = HomFGA(A, target = target)

    def test_project_to_source(self):
        """
        Test the periods.
        """

        # Project to source
        phi = self.phi.project_to_source()

        # Retrieve the periods as a vector
        periods = Matrix(phi.source.periods)

        # Assert that the columns times the periods are zero
        assert phi.evaluate(periods) == Matrix(phi.target.periods) * 0


    def test_image_coimage_factorization(self):
        """
        Test the image/coimage factorization.
        """

        # Compute the image morphism and the coimage morphism
        image = self.phi.image().remove_trivial_groups()
        coimage = self.phi.coimage().remove_trivial_groups()

        # Asser that the coimage/image factorization holds
        factorization = (image * coimage).project_to_target()
        original = (self.phi).project_to_target()
        assert factorization == original

    def test_kernel(self):
        """
        The that the kernel and the morphism is zero.
        """
        # Compute the kernel
        kernel = self.phi.kernel()
        phi_ker = (self.phi * kernel).project_to_target()
        zero_morphism = HomFGA.zero(target = self.phi.target,
                                    source = kernel.source)
        assert phi_ker == zero_morphism

    def test_cokernel(self):
        """
        Test that the composition of the cokernel and the morphism is zero.
        """
        cokernel = self.phi.cokernel()
        coker_phi = (cokernel * self.phi).project_to_target()
        zero_morphism = HomFGA.zero(source = self.phi.source,
                                    target = cokernel.target)
        assert coker_phi == zero_morphism

if __name__ == '__main__':

    t = TestSNF()
    t.setup_class()
    t.test_project_to_source()
    t.test_image_coimage_factorization()
    t.test_kernel()
    t.test_cokernel()