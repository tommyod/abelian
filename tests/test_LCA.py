#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import reduce
from operator import add
from random import randint as ri
from random import shuffle

import pytest

from abelian.groups import LCA
from utils import random_LCA


class TestLCA:

    @pytest.fixture
    def random_LCAs(self):
        G = random_LCA(10)
        H = random_LCA(10)

        return G, H

    def test_border_cases(self):
        """ Border cases when initializing """
        identity = LCA.trivial()
        assert identity.isomorphic(LCA([]))

    def test_canonical(self, random_LCAs):
        """ Test that canonical is invariant under shuffling """
        G, _ = random_LCAs
        G_split = [grp for grp in G]

        shuffle(G_split)
        G_shuffled = reduce(add, G_split)

        assert G.canonical() == G_shuffled.canonical()

    def test_trivial_group(self, random_LCAs):
        """ Test the property of the trivial group """
        G, H = random_LCAs

        identity = LCA.trivial()
        assert (identity.compose_self(ri(0, 3))).isomorphic(identity)
        assert (G + identity).isomorphic(G)
        assert (identity + G).isomorphic(G)
        assert (H + identity).isomorphic(H)
        assert (identity + H).isomorphic(H)
        assert (identity + identity).isomorphic(identity)

    def test_rank(self, random_LCAs):
        """ Test the rank of LCAs """
        G, H = random_LCAs

        assert (G + H).rank() == (G.rank() + H.rank())

    def test_length(self, random_LCAs):
        """ Test the length of LCAs """
        G, H = random_LCAs

        assert (G + H).length() == (G.length() + H.length())

    def test_remove_trivial(self, random_LCAs):
        """ Test the removal of trivial subgroups """
        G, H = random_LCAs
        after = (G + H).remove_trivial()
        before = (G.remove_trivial() + H.remove_trivial())

        assert before == after

    def test_proper_inclusion(self, random_LCAs):
        """ Test proper inclusion """
        G, _ = random_LCAs
        H = G[3:-3]

        assert (H in G)

    def test_inclusion(self, random_LCAs):
        """ Test inclusion """
        G, H = random_LCAs

        assert (G in G) and (H in H)
