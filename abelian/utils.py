#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import functools

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


argmin = functools.partial(arg, min_or_max = min)
argmax = functools.partial(arg, min_or_max = max)

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = True)