#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import itertools
import random
import types

import numpy as np
from sympy import Integer, Float, Rational


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


def verify_dims_list(list_of_lists, dims):
    """
    Verify the dimensions of a list of lists.

    Parameters
    ----------
    list_of_lists : list
        A nested list of lists.
    dims : list
        A list of dimensions.

    Returns
    -------
    verified : bool
        Whether or not the dimensions match the `dims` parameter.

    Examples
    ---------
    >>> table = [1 ,2 ,3]
    >>> dims = [3]
    >>> verify_dims_list(table, dims)
    True

    >>> table = [[1, 2], [1, 2], [1, 2]]
    >>> dims = [3, 2]
    >>> verify_dims_list(table, dims)
    True

    >>> table = [[[1, 2, 3, 4], [1, 2, 3, 4]],
    ...          [[1, 2, 3, 4], [1, 2, 4]],
    ...          [[1, 2, 3, 4], [1, 2, 3, 4]]]
    >>> dims = [3, 2, 4] # Not correct, notice the missing value above
    >>> verify_dims_list(table, dims)
    False
    """
    for i, dim in enumerate(dims):
        product_arg = [range(dims[k]) for k in range(i)]
        for p in itertools.product(*product_arg):
            if not len(call_nested_list(list_of_lists, p)) == dim:
                return False

    return True


def call_nested_list(list_of_lists, arg):
    """
    Call a nested list like a function.

    Parameters
    ----------
    list_of_lists : list
        A nested list of lists.
    arg : list
        The argument [dim1, dim2, ...].

    Returns
    -------
    value : object
        The object in the list of lists.

    Examples
    ---------
    >>> table = [1 ,2 ,3]
    >>> call_nested_list(table, [0])
    1

    >>> table = [[1, 2], [1, 2], [1, 2]]
    >>> call_nested_list(table, [0, 0])
    1

    >>> table = [[[1, 2, 3, 4], [1, 2, 3, 4]],
    ...          [[1, 2, 3, 4], [1, 2, 3, 4]],
    ...          [[1, 2, 3, 4], [1, 2, 3, 4]]]
    >>> call_nested_list(table, [0, 0])
    [1, 2, 3, 4]
    >>> call_nested_list(table, [0, 0, 0])
    1
    """

    answer = list_of_lists
    for index in arg:
        answer = answer[index]
    return answer


def function_to_table(function, dims, *args, **kwargs):
    """

    Parameters
    ----------
    function : function
        A function with signature `(list_arg, *args, **kwargs)`.
    dims : list
        A list of dimensions such as [8, 6, 3].

    Returns
    -------

    """

    # Create an empty table using numpy
    table = np.empty(dims, dtype=complex)

    # Iterate through the domain and evaluate the function
    for list_arg in itertools.product(*[range(d) for d in dims]):
        table[list_arg] = function(list(list_arg),  *args, **kwargs)

    return table


def arg(min_or_max, iterable, function_of_element):
    """
    Call a nested list like a function.

    Parameters
    ----------
    list_of_lists : list
        A nested list of lists.
    arg : list
        The argument [dim1, dim2, ...].

    Returns
    -------
    value : object
        The object in the list of lists.

    Examples
    ---------
    >>> iterable = [-8, -4, -2, 3, 5]
    >>> arg(min, iterable, abs)
    -2
    >>> iterable = range(-10, 10)
    >>> arg(max, iterable, lambda x: -(x - 3)**2)
    3
    """
    return min_or_max(iterable, key=function_of_element)


def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


argmin = functools.partial(arg, min_or_max=min)
argmax = functools.partial(arg, min_or_max=max)


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


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
