#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from random import randint as ri
from random import shuffle
from abelian.groups import LCA
from functools import reduce
from operator import add

def random_zero_heavy(low, high):
    """
    Draw a random number, with approx 50% probability of zero.
    """
    return random.choice(list(range(low, high)) + [0]*(high - low))

def random_from_list(number, list_to_take_from):
    """
    Draw several random values from the same list.
    """
    return [random.choice(list_to_take_from) for i in range(number)]

def random_LCA(length):
    """
    Create a random LCA of a given length.
    """
    discrete = random_from_list(length, [True, False])
    periods = []
    for d in discrete:
        if d:
            # Discrete group, any integer
            periods.append(random_zero_heavy(0, 99))
        else:
            periods.append(random.choice([0, 1]))

    return LCA(orders= periods, discrete = discrete)


class TestLCA:

    @classmethod
    def setup_class(cls):
        cls.G = random_LCA(10)
        cls.H = random_LCA(10)

    def test_border_cases(self):
        """
        Border cases when initializing.
        """
        Id = LCA.trivial()

        assert Id.isomorphic(LCA([]))

    def test_canonical(self):
        """
        Test that canonical is invariant under shuffling.
        """
        G = self.G

        G_split = [grp for grp in G]
        shuffle(G_split)
        G_shuffled = reduce(add, G_split)

        assert G.canonical() == G_shuffled.canonical()

    def test_trivial_group(self):
        """
        Test the property of the trivial group.
        """
        Id = LCA.trivial()
        G = self.G
        H = self.H
        assert (G + Id).isomorphic(G)
        assert (Id + G).isomorphic(G)
        assert (H + Id).isomorphic(H)
        assert (Id + H).isomorphic(H)
        assert (Id + Id).isomorphic(Id)


    def test_rank(self):
        """
        Test the rank of LCAs.
        """
        G = self.G
        H = self.H
        assert (G + H).rank() == (G.rank() + H.rank())

    def test_length(self):
        """
        Test the length of LCAs.
        """
        G = random_LCA(ri(1, 9))
        H = random_LCA(ri(1, 9))
        assert (G + H).length() == (G.length() + H.length())

    def test_remove_trivial(self):
        """
        Test the removal of trivial subgroups.
        """
        G = self.G
        H = self.H
        after = (G + H).remove_trivial()
        before = (G.remove_trivial() + H.remove_trivial())
        assert before == after

    def test_proper_subgroup(self):
        """
        Test proper subgroups.
        """
        G = self.G
        H = G[3:-3]
        assert (H in G)

    def test_subgroup(self):
        """
        Test subgroups.
        """
        G = self.G
        assert (G in G)



if __name__ == '__main__':
    from abelian import LCA, HomLCA, LCAFunc, voronoi
    from math import exp, pi
    Z = LCA(orders = [0], discrete = [True])
    R = LCA(orders = [0], discrete = [False])

    # Create the Gaussian function on R^2
    function = LCAFunc(lambda x: exp(-pi*sum(j**2 for j in x)), domain = R**2)

    # Create an orthogonal sampling homomorphism
    phi = HomLCA([[0.05, 0], [0, 0.05]], source = Z**2, target = R**2)
    function_sampled = function.pullback(phi)

    # Approximate the two-dimensional integral of the gaussian
    scaling_factor = phi.A.det()
    integral_value = 0
    for element in phi.source.elements_by_maxnorm(list(range(50))):
        integral_value += function_sampled(element)
    print(integral_value * scaling_factor) # 0.999999998926396

    # Sample, periodize and take DFT of the Gaussian
    phi_p = HomLCA([[15, 0], [0, 15]], source = Z**2, target = Z**2)
    periodized = function_sampled.pushforward(phi_p.cokernel())
    dual_func = periodized.dft()
    DFT_ouput = dual_func.table * phi_p.A.det() * scaling_factor

    # Interpret the output of the DFT on R^2
    phi_periodize_ann = phi_p.annihilator()

    # Compute a Voronoi transversal function, interpret on R**2
    sigma = voronoi(phi.dual(), norm_p=2)
    for element in dual_func.domain.elements_by_maxnorm():
        value = dual_func(element)
        coords_on_R = sigma(phi_periodize_ann(element))

        # The function is invariant under Fourier transformation, so we can
        # compare the error analytically
        true_val = function(coords_on_R) # The function is invariant under FT
        approximated_val = abs(value)
        assert abs(true_val - approximated_val) < 0.01

