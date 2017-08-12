#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import randint as ri
import random
from sympy import Matrix, pprint
from abelian.linalg.utils import vector_mod_vector, remove_cols
from abelian.linalg.solvers import solve, solve_epi
from abelian.linalg.factorizations import hermite_normal_form


class TestSolveEpi():

    @classmethod
    def setup_class(cls):
        # Set up the sizes for the tests
        cls.m = 4
        cls.k = 2


    def test_bijective_free_to_free(self):
        """
        Solve X*A = B when A is a bijective, free-to-free morphism.
        """
        m, k = self.m, self.k
        A = Matrix(m, m, lambda i, j : ri(-9, 9))
        A, H = hermite_normal_form(A)
        X = Matrix(k, m, lambda i, j : ri(-9, 9))
        B = X * A

        X_sol = solve_epi(A, B)
        assert B == X_sol * A


    def test_epi_free_to_free(self):
        """
        Solve X*A = B when A is overdetermined, free-to-free morphism.
        """
        m, k = self.m, self.k

        # Create a matrix A
        A = Matrix(m, m, lambda i, j : ri(-9, 9))
        A, H = hermite_normal_form(A)
        extra_cols = ri(1, 3)

        # Add extra columns, linear combination of existing
        lin_combs = Matrix(m, extra_cols, lambda i, j: ri(-9, 9))
        A = A * (Matrix.eye(m).row_join(lin_combs))
        X = Matrix(k, m, lambda i, j : ri(-9, 9))
        B = X * A

        X_sol = solve_epi(A, B)
        assert B == X_sol * A


class TestSolve():

    @classmethod
    def setup_class(cls):
        # Set up the sizes for the tests
        cls.n = 4
        cls.r = cls.n - 2


    def test_bijective(self):
        """
        Test the equation solver when A is n x n and of full rank.
        """

        # Set up the equation
        n = self.n
        A = Matrix(n, n, lambda i,j : ri(-9,9))
        A, H = hermite_normal_form(A)
        x = Matrix(n, 1, lambda i, j : ri(-5,5))
        p = Matrix(n, 1, lambda i, j : ri(10,100))
        b = vector_mod_vector(A*x, p)

        # Solve the equation and verify
        x_sol = solve(A, b, p)
        assert (vector_mod_vector(A*x_sol, p) == b)

    def test_surjective(self):
        """
        Test the equation solver when A is surjective,
        i.e. maps onto (epimorphism).
        """

        # Create a matrix A
        n = self.n
        A = Matrix(n, n, lambda i, j: ri(-9, 9))
        A, H = hermite_normal_form(A)
        m, m = A.shape
        extra_cols = ri(1, 3)

        # Add extra columns, linear combinations of existing columns
        A = A * (Matrix.eye(m).row_join(Matrix(m, extra_cols, lambda i,j :ri(-9, 9))))
        p = Matrix(m, 1, lambda i, j: ri(10, 100))
        x = Matrix(m + extra_cols, 1, lambda i, j: ri(-5, 5))
        b = vector_mod_vector(A * x, p)

        x_sol = solve(A, b, p)
        assert (vector_mod_vector(A * x_sol, p) == b)

    def test_injective(self):
        """
        Test the equation solver when a is surjective,
        i.e. maps one-to-one (monomorphism).
        """

        # Create a unimodular matrix A
        n = self.n
        A = Matrix(n, n, lambda i, j: ri(-9, 9))
        A, H = hermite_normal_form(A)
        m, m = A.shape
        num_cols_remove = ri(1, m-1)

        # Remove columns from A
        cols_to_del = random.sample([i for i in range(m)], num_cols_remove)
        A = remove_cols(A, cols_to_del)

        m, n = A.shape
        p = Matrix(m, 1, lambda i, j: ri(2, 99))
        x = Matrix(n, 1, lambda i, j: ri(-5, 5))

        b = vector_mod_vector(A * x, p)

        x_sol = solve(A, b, p)
        assert (vector_mod_vector(A * x_sol, p) == b)

    def test_lower_rank_mapping(self):
        """
        Test the equation solver on a matrix A
        which is neither surjective nor injective.
        """
        # Create a matrix of lower rank
        n = self.n
        r = self.r
        g = Matrix(r, n, lambda i, j : ri(-9, 9))
        A = g.T * g

        # Set up equations
        x = Matrix(n, 1, lambda i, j: ri(-5, 5))
        p = Matrix(n, 1, lambda i, j: ri(10, 100))
        b = vector_mod_vector(A * x, p)

        # Solve the equation and verify
        x_sol = solve(A, b, p)
        assert (vector_mod_vector(A * x_sol, p) == b)


