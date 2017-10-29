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
