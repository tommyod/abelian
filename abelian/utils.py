#!/usr/bin/env python
# -*- coding: utf-8 -*-

def mod(a, b):
    """
    Returns a % b, with a % 0 = a.

    Parameters
    ----------
    a : int
        The first argument in a % b.
    b : int
        The second argument in a % b.

    Returns
    -------
    int
        `a` modulus `b`.

    Examples
    ---------
    >>> mod(5, 2)
    1
    >>> mod(5, 0)
    5
    """
    if b == 0:
        return a
    return a % b




if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = True)