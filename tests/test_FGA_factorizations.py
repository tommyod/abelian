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

class TestSNF:

    @staticmethod
    def setup():
        """
        Setup a homomorphism.
        """

        # --------- START SETUP ---------
        m, n = ri(1, 5), ri(1, 5)
        A = Matrix(m, n, lambda i, j: random_zero_heavy(-9, 9))
        target = Matrix(m, 1, lambda i, j: random_zero_heavy(5, 50))
        target_vector = target
        phi = HomLCA(A, target = target)
        # --------- END SETUP ---------
        return phi, target_vector


    def test_source_projection_two_ways(self):
        """
        Test that phi.project to source == phi.coimage.project_to_source
        """
        phi, target_vector = self.setup()

        # Project to source
        phi = phi.project_to_source()
        coimage = phi.coimage().project_to_source()
        assert phi.source == coimage.source


    def test_project_to_source(self):
        """
        Test the orders.
        """
        phi, target_vector = self.setup()

        # Project to source
        phi = phi.project_to_source()

        # Retrieve the orders as a vector
        orders = Matrix(phi.source.orders)

        # Assert that the columns times the orders are zero
        assert phi.evaluate(orders) == Matrix(phi.target.orders) * 0


    def test_image_coimage_factorization(self):
        """
        Test the image/coimage factorization.
        """
        phi, target_vector = self.setup()

        # Compute the image morphism and the coimage morphism
        image = phi.image().remove_trivial_groups()
        coimage = phi.coimage().remove_trivial_groups()

        # Asser that the coimage/image factorization holds
        factorization = (image * coimage).project_to_target()
        original = (phi).project_to_target()
        assert factorization == original


    def test_kernel(self):
        """
        The that the kernel and the morphism is zero.
        """
        phi, target_vector = self.setup()

        # Compute the kernel
        kernel = phi.kernel()
        phi_ker = (phi * kernel).project_to_target()
        zero_morphism = HomLCA.zero(target = phi.target,
                                    source = kernel.source)
        assert phi_ker == zero_morphism


    def test_cokernel(self):
        """
        Test that the composition of the cokernel and the morphism is zero.
        """
        phi, target_vector = self.setup()

        cokernel = phi.cokernel()
        coker_phi = (cokernel * phi).project_to_target()
        zero_morphism = HomLCA.zero(source = phi.source,
                                    target = cokernel.target)
        assert coker_phi == zero_morphism