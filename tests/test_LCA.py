#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from functools import reduce
from operator import add
from random import randint as ri
from random import shuffle

import pytest

from abelian.groups import LCA
from utils import random_zero_heavy, random_from_list


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

    return LCA(orders=periods, discrete=discrete)


@pytest.fixture
def setup():
    G = random_LCA(10)
    H = random_LCA(10)

    return G, H


def test_border_cases():
    """
    Border cases when initializing.
    """
    Id = LCA.trivial()

    assert Id.isomorphic(LCA([]))


def test_canonical(setup):
    """
    Test that canonical is invariant under shuffling.
    """
    G, H = setup

    G_split = [grp for grp in G]
    shuffle(G_split)
    G_shuffled = reduce(add, G_split)

    assert G.canonical() == G_shuffled.canonical()


def test_trivial_group(setup):
    """
    Test the property of the trivial group.
    """
    G, H = setup

    Id = LCA.trivial()
    assert (Id.compose_self(ri(0, 3))).isomorphic(Id)
    assert (G + Id).isomorphic(G)
    assert (Id + G).isomorphic(G)
    assert (H + Id).isomorphic(H)
    assert (Id + H).isomorphic(H)
    assert (Id + Id).isomorphic(Id)


def test_rank(setup):
    """
    Test the rank of LCAs.
    """
    G, H = setup

    assert (G + H).rank() == (G.rank() + H.rank())


def test_length(setup):
    """
    Test the length of LCAs.
    """
    G, H = setup

    assert (G + H).length() == (G.length() + H.length())


def test_remove_trivial(setup):
    """
    Test the removal of trivial subgroups.
    """
    G, H = setup

    after = (G + H).remove_trivial()
    before = (G.remove_trivial() + H.remove_trivial())
    assert before == after


def test_proper_inclusion(setup):
    """
    Test proper inclusion.
    """
    G, _ = setup

    H = G[3:-3]
    assert (H in G)


def test_inclusion(setup):
    """
    Test inclusion.
    """
    G, H = setup

    assert (G in G)
    assert (H in H)
