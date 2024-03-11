#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from random import randint as ri
from random import random

import pytest
from sympy import Matrix

from abelian.morphisms import HomLCA, LCA
from abelian.utils import frob_norm


@pytest.fixture
def setup_homomorphism():
    m, n = ri(2, 5), ri(2, 5)
    A = Matrix(m, n, lambda i, j: random())

    # Add an extra column to A
    new_col = A[:, 0] + A[:, 1]
    A = A.row_join(new_col)

    # Add an extra row to A
    new_row = A[0, :] + A[1, :]
    A = A.col_join(new_row)

    R = LCA([0], [False])
    return HomLCA(A, source=R ** (n + 1), target=R ** (m + 1))


def test_real_kernel(setup_homomorphism):
    """
    Test the real kernel.
    """
    phi = setup_homomorphism

    phi_ker = phi.kernel()

    A = (phi * phi_ker).A
    B = HomLCA.zero(target=phi.target, source=phi_ker.source).A
    assert frob_norm(A, B) < 10e-10


def test_real_cokernel(setup_homomorphism):
    """
    Test the real cokernel.
    """
    phi = setup_homomorphism

    phi_coker = phi.cokernel()

    A = (phi_coker * phi).A
    B = HomLCA.zero(target=phi_coker.target, source=phi.source).A
    assert frob_norm(A, B) < 10e-10


def test_real_im_coim(setup_homomorphism):
    """
    Test the real image/coimage.
    """
    phi = setup_homomorphism

    phi_image = phi.image()
    phi_coimage = phi.coimage()

    B = (phi_image * phi_coimage).A
    assert frob_norm(phi.A, B) < 10e-10
