#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from random import randint as ri
from abelian.utils import elements_of_maxnorm



class TestElementsGenerator:

    def test_elements_of_maxnorm_num_elements(self):
        """
        Verify the number of elements.
        """

        # Random parameter values
        dim = ri(1, 5)
        normvalue = ri(3, 5)

        # Theoretical value
        theoretical_value = (2*normvalue + 1)**dim - (2*normvalue - 1)**dim

        # Actual value
        generated = list(elements_of_maxnorm(dim, normvalue))
        assert len(set(generated)) == theoretical_value


    def test_elements_of_maxnorm_increasing_norm(self):
        """
        Verify that the norm is increasing.
        """

        # Random parameter values
        dim = ri(1, 4)
        normvalue = ri(2, 4)

        # Generate elements up to a norm
        generated = []
        for k in range(normvalue):
            generated += list(elements_of_maxnorm(dim, k))

        # Define the maximum norm
        norm = lambda v : max(abs(k) for k in v)

        # Assert that the norm is increasing
        assert all(norm(a)<= norm(b) for a, b in
                   zip(generated[:-1], generated[1:]))




