#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import exp, sqrt, pi
from random import choice
from random import randint

from sympy import Matrix

from abelian import HomLCA, LCA
from abelian import LCAFunc, voronoi
from utils import close


def test_example_1_HomFGA():
    """
    Test example 1 from the abelian introductory chapter
    in the thesis.
    """

    target = LCA([8, 5])  # Create Z_8 + Z_5
    phi = HomLCA([[4, 2], [7, 3]], target=target)

    # Compute cokernel, then remove trivial groups
    cokernel = phi.cokernel().remove_trivial_groups()

    # Compute image, then remove trivial groups
    image = phi.image().remove_trivial_groups()

    # Compute coimage, remove trivial, then project
    coimage = phi.coimage().remove_trivial_groups()
    coimage = coimage.project_to_source()

    # Project phi, compute kernel
    phi_projected = phi.project_to_source()
    kernel = phi_projected.kernel().project_to_source()

    assert cokernel.A == Matrix([[1, 0]])
    assert image.A == Matrix([[2], [3]])
    assert coimage.A == Matrix([[14, 1]])
    assert kernel.A == Matrix([[2, 5], [12, 10]])


def test_example_2_FourierSeries():
    """
    Test example 2 from the abelian introductory chapter
    in the thesis.
    """
    # Import objects, create function on T^2
    from abelian import HomLCA, LCA, LCAFunc
    from sympy import Rational, diag
    T = LCA(orders=[1], discrete=[False])
    func = LCAFunc(lambda x: sum(x), domain=T ** 2)

    # Create homomorphism to sample function
    n = randint(6, 10)
    Z_n = LCA(orders=[n], discrete=[True])
    phi = HomLCA(diag(Rational(1, n), Rational(1, n)),
                 target=T ** 2, source=Z_n ** 2)

    # Sample, dualize
    func_sampled = func.pullback(phi)
    func_sample_dual = func_sampled.dft()

    # Transversal - minimizes distance
    func_dual = func_sample_dual.transversal(phi.dual())

    assert func([0.5, 0.5]) == 1
    assert close(func([1.6, 0.6]), 1.2)
    assert close(func_sampled([n + 1, n + 1]), func([1 / n, 1 / n]))
    assert close(func_sampled([1, 1]), func([1 / n, 1 / n]))

    assert func_dual([11, 11]) == 0
    assert close(func_dual([0, 0]), (1 - 1 / n) * n * n)


def test_example_3_Hexagonal():
    # create function on R^n
    R = LCA(orders=[0], discrete=[False])
    k = choice([0.85, 0.9, 1, 1.1, 1.15])  # Decay of exponential
    func_expr = lambda x: exp(-pi * k * sum(x_j ** 2 for x_j in x))
    func = LCAFunc(func_expr, domain=R ** 2)

    # Create a homomorphism to sample
    hexagonal_generators = [[1, 0.5], [0, sqrt(3) / 2]]
    phi_sample = HomLCA(hexagonal_generators, target=R ** 2)
    phi_sample = phi_sample * (1 / 7)

    # Create a homomorphism to periodize
    n = 3
    phi_periodize = HomLCA([[n, 0], [0, n]])
    coker_phi_p = phi_periodize.cokernel()

    # Move function from R**2 to Z**2 to Z_n**2
    func_sampled = func.pullback(phi_sample)
    func_periodized = func_sampled.pushforward(coker_phi_p, 25)

    # Move function to dual space, then to T**2
    func_dual = func_periodized.dft()
    phi_periodize_ann = phi_periodize.annihilator()

    scale_factor = phi_sample.A.det()

    # Compute a Voronoi transversal function, interpret on R**2
    sigma = voronoi(phi_sample.dual(), norm_p=2)
    for element in func_dual.domain.elements_by_maxnorm():
        value = func_dual(element)
        coords_on_R = sigma(phi_periodize_ann(element))

        approx_value = abs(value) * scale_factor
        true_value = func(coords_on_R)

        assert abs(approx_value - true_value / k) < 0.10
