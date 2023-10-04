import random

from sympy import Integer, Float, Rational


def close(a, b):
    numeric_types = (float, int, complex, Integer, Float, Rational)
    if isinstance(a, numeric_types) and isinstance(a, numeric_types):
        return abs(a - b) < 10e-10
    return sum(abs(i - j) for (i, j) in zip(a, b))


def random_zero_heavy(low, high):
    """
    Draw a random number, with approx 50% probability of zero.
    """
    return random.choice(list(range(low, high)) + [0] * (high - low))


def frob_norm(A, B):
    """
    Frobenius norm.
    """
    return sum(abs(i - j) for (i, j) in zip(A, B))


def random_from_list(number, list_to_take_from):
    """
    Draw several random values from the same list.
    """
    return [random.choice(list_to_take_from) for i in range(number)]
