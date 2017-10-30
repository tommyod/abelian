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

    @staticmethod
    def setup():
        G = random_LCA(10)
        H = random_LCA(10)

        return G, H

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
        G, H = self.setup()

        G_split = [grp for grp in G]
        shuffle(G_split)
        G_shuffled = reduce(add, G_split)

        assert G.canonical() == G_shuffled.canonical()

    def test_trivial_group(self):
        """
        Test the property of the trivial group.
        """
        G, H = self.setup()

        Id = LCA.trivial()
        assert (Id.compose_self(ri(0,3))).isomorphic(Id)
        assert (G + Id).isomorphic(G)
        assert (Id + G).isomorphic(G)
        assert (H + Id).isomorphic(H)
        assert (Id + H).isomorphic(H)
        assert (Id + Id).isomorphic(Id)


    def test_rank(self):
        """
        Test the rank of LCAs.
        """
        G, H = self.setup()

        assert (G + H).rank() == (G.rank() + H.rank())

    def test_length(self):
        """
        Test the length of LCAs.
        """
        G, H = self.setup()

        assert (G + H).length() == (G.length() + H.length())

    def test_remove_trivial(self):
        """
        Test the removal of trivial subgroups.
        """
        G, H = self.setup()

        after = (G + H).remove_trivial()
        before = (G.remove_trivial() + H.remove_trivial())
        assert before == after

    def test_proper_inclusion(self):
        """
        Test proper inclusion.
        """
        G, _ = self.setup()

        H = G[3:-3]
        assert (H in G)

    def test_inclusion(self):
        """
        Test inclusion.
        """
        G, H = self.setup()

        assert (G in G)
        assert (H in H)