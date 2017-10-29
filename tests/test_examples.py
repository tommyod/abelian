#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from abelian import LCA, LCAFunc, HomLCA, voronoi
from sympy import Matrix, diag, Integer, Float, Rational


def close(a, b):
    numeric_types = (float, int, complex, Integer, Float, Rational)
    if isinstance(a, numeric_types) and isinstance(a, numeric_types):
        return abs(a-b) < 10e-10
    return sum(abs(i-j) for (i, j) in zip(a, b))

class TestThesisExamples:

    def test_example_1_HomFGA(self):
        """
        Test example 1 from the abelian introductory chapter
        in the thesis.
        """

        # Import the classes and create a homomorphism
        from abelian import HomLCA, LCA
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

    def test_example_2_FourierSeries(self):
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
        n = 10
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
        assert close(func_sampled([11, 11]), func([0.1, 0.1]))
        assert close(func_sampled([1, 1]),   func([0.1, 0.1]))

        assert func_dual([11, 11]) == 0
        assert close(func_dual([0, 0]), 0.9)

    def test_example_3_Hexagonal(self):
        from math import exp, sqrt, pi

        # Import objects, create function on R^n
        from abelian import HomLCA, LCA, LCAFunc, voronoi
        R = LCA(orders=[0], discrete=[False])
        k = 0.5  # Decay of exponential
        func_expr = lambda x: exp(-k * sum(x_j ** 2 for x_j in x))
        func = LCAFunc(func_expr, domain=R ** 2)

        # Create a homomorphism to sample
        hexagonal_generators = [[1, 0.5], [0, sqrt(3) / 2]]
        phi_sample = HomLCA(hexagonal_generators, target=R ** 2)

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

        # Compute a Voronoi transversal function, interpret on R**2
        sigma = voronoi(phi_sample.dual(), norm_p=2)
        for element in func_dual.domain.elements_by_maxnorm():
            value = func_dual(element)
            coords_on_R = sigma(phi_periodize_ann(element))

        approx, true = func_dual([0, 0])*n*n, sqrt((2*pi)**2)
        print(approx, true, true/approx)

        print(phi_sample.A.det()*2)

        print(approx*phi_sample.A.det()*2, true/k)


if __name__ == '__main__':
    t = TestThesisExamples()
    t.test_example_3_Hexagonal()