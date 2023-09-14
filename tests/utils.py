from random import choice
from abelian.groups import LCA


def random_zero_heavy(low, high):
    """
    Draw a random number, with approx 50% probability of zero.
    """
    return choice(list(range(low, high)) + [0] * (high - low))


def random_from_list(number, list_to_take_from):
    """
    Draw several random values from the same list.
    """
    return [choice(list_to_take_from) for _ in range(number)]


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
            periods.append(choice([0, 1]))

    return LCA(orders=periods, discrete=discrete)
