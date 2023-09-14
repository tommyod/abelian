#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import randint as ri

import pytest
from sympy import Matrix

from abelian.morphisms import HomLCA
from utils import random_zero_heavy


class TestSNF:

    @pytest.fixture
    def homomorphism(self):
        """ Set up a homomorphism """
        m, n = ri(1, 5), ri(1, 5)
        A = Matrix(m, n, lambda i, j: random_zero_heavy(-9, 9))
        target = Matrix(m, 1, lambda i, j: random_zero_heavy(5, 50))
        target_vector = target
        phi = HomLCA(A, target=target)
        return phi, target_vector

    def test_source_projection_two_ways(self, homomorphism):
        """ Test that phi.project_to_source == phi.coimage.project_to_source """
        phi, target_vector = homomorphism

        # Project to source
        phi = phi.project_to_source()
        coimage = phi.coimage().project_to_source()

        assert phi.source == coimage.source

    def test_project_to_source(self, homomorphism):
        """ Test the orders """
        phi, target_vector = homomorphism

        # Project to source
        phi = phi.project_to_source()

        # Retrieve the orders as a vector
        orders = Matrix(phi.source.orders)

        # Assert that the columns times the orders are zero
        assert phi.evaluate(orders) == Matrix(phi.target.orders) * 0

    def test_image_coimage_factorization(self, homomorphism):
        """ Test the image/coimage factorization """
        phi, target_vector = homomorphism

        # Compute the image morphism and the coimage morphism
        image = phi.image().remove_trivial_groups()
        coimage = phi.coimage().remove_trivial_groups()

        # Asser that the coimage/image factorization holds
        factorization = (image * coimage).project_to_target()
        original = phi.project_to_target()

        assert factorization == original

    def test_kernel(self, homomorphism):
        """ The that the kernel and the morphism is zero """
        phi, target_vector = homomorphism

        # Compute the kernel
        kernel = phi.kernel()
        phi_ker = (phi * kernel).project_to_target()
        zero_morphism = HomLCA.zero(target=phi.target, source=kernel.source)

        assert phi_ker == zero_morphism

    def test_cokernel(self, homomorphism):
        """ Test that the composition of the cokernel and the morphism is zero """
        phi, target_vector = homomorphism

        cokernel = phi.cokernel()
        cokernel_phi = (cokernel * phi).project_to_target()
        zero_morphism = HomLCA.zero(source=phi.source, target=cokernel.target)

        assert cokernel_phi == zero_morphism
