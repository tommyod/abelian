#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import randint as ri

import pytest
from sympy import Matrix

from abelian.morphisms import HomLCA
from abelian.utils import random_zero_heavy


@pytest.fixture
def setup():
    """
    Set up a homomorphism.
    """
    m, n = ri(1, 5), ri(1, 5)
    A = Matrix(m, n, lambda i, j: random_zero_heavy(-9, 9))
    target = Matrix(m, 1, lambda i, j: random_zero_heavy(5, 50))
    target_vector = target
    phi = HomLCA(A, target=target)

    return phi, target_vector


def test_source_projection_two_ways(setup):
    """
    Test that phi.project to source == phi.coimage.project_to_source
    """
    phi, _ = setup

    # Project to source
    phi = phi.project_to_source()
    coimage = phi.coimage().project_to_source()
    assert phi.source == coimage.source


def test_project_to_source(setup):
    """
    Test the orders.
    """
    phi, _ = setup

    # Project to source
    phi = phi.project_to_source()

    # Retrieve the orders as a vector
    orders = Matrix(phi.source.orders)

    # Assert that the columns times the orders are zero
    assert phi.evaluate(orders) == Matrix(phi.target.orders) * 0


def test_image_coimage_factorization(setup):
    """
    Test the image/coimage factorization.
    """
    phi, _ = setup

    # Compute the image morphism and the coimage morphism
    image = phi.image().remove_trivial_groups()
    coimage = phi.coimage().remove_trivial_groups()

    # Asser that the coimage/image factorization holds
    factorization = (image * coimage).project_to_target()
    original = (phi).project_to_target()
    assert factorization == original


def test_kernel(setup):
    """
    The that the kernel and the morphism is zero.
    """
    phi, _ = setup

    # Compute the kernel
    kernel = phi.kernel()
    phi_ker = (phi * kernel).project_to_target()
    zero_morphism = HomLCA.zero(target=phi.target,
                                source=kernel.source)
    assert phi_ker == zero_morphism


def test_cokernel(setup):
    """
    Test that the composition of the cokernel and the morphism is zero.
    """
    phi, target_vector = setup

    cokernel = phi.cokernel()
    coker_phi = (cokernel * phi).project_to_target()
    zero_morphism = HomLCA.zero(source=phi.source,
                                target=cokernel.target)
    assert coker_phi == zero_morphism
