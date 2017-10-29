#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module consists of a class for elementary locally compact abelian groups,
the LCA class.
"""

from sympy import Matrix, Integer, diag
from abelian.utils import mod
from abelian.linalg.factorizations import smith_normal_form
from abelian.linalg.utils import nonzero_diag_as_list
from collections.abc import Sequence, Callable
from abelian.linalg.free_to_free import elements_of_maxnorm_FGA
import itertools

class LCA(Sequence, Callable):
    """
    An elementary locally compact abelian group (LCA).
    """

    # These dictionaries lets the user initialize using
    _repr_dict = {True: 'd', False :'c'}
    _repr_dict_inv = {'d':True, 'c':False}

    _integer_types = (int, Integer)

    def __init__(self, orders, discrete = None):
        """
        Initialize a new LCA.

        This class represents locally compact abelian groups, defined by their
        orders and whether or not they are discrete. An order of 0 means
        infinite order. The possible groups are:

        * :math:`\mathbb{Z}_n` : order = `n`, discrete = `True`
        * :math:`\mathbb{Z}` : order = `0`, discrete = `True`
        * :math:`T` : order = `1`, discrete = `False`
        * :math:`\mathbb{R}` : order = `0`, discrete = `False`

        Every locally compact abelian group is isomorphic to a direct sum
        or one or several of the groups above.

        Parameters
        ----------
        orders : list
            A list of orders, e.g. [6, 8, 11].
        discrete : list
            A list of booleans such as [True, False, ...] or alternatively a
            list of letters such as ['d', 'c', ...], where 'd' stands for
            discrete and 'c' stands for continuous. If None, it defaults to
            discrete.


        Examples
        ---------
        >>> # Create G = Z_5 + Z_6 + Z_7 in three ways
        >>> G1 = LCA([5, 6, 7])
        >>> G2 = LCA([5, 6, 7], [True, True, True])
        >>> G3 = LCA([5, 6, 7], ['d']*3)
        >>> (G1 == G2 == G3)
        True

        >>> # Create G = R + Z
        >>> G = LCA(orders = [0, 0], discrete = [False, True])

        >>> G = LCA([], [])
        >>> G
        []
        """

        orders, discrete = self._verify_init(orders, discrete)
        self.orders = orders
        self.discrete = discrete


    @classmethod
    def trivial(cls):
        """
        Return a trivial LCA.

        Returns
        --------
        group : LCA
            A trivial LCA.

        Examples
        ---------
        >>> trivial = LCA.trivial()
        >>> Z = LCA([0])
        >>> (Z + trivial).isomorphic(Z)
        True
        """
        return cls(orders = [1], discrete = [True])


    @classmethod
    def _verify_init(cls, orders, discrete):
        """
        Verify the user inputs.

        Return `orders` and `discrete` as lists.
        """

        # Forgive if integer types have been passed, or if it's a matrix
        if isinstance(orders, (Matrix,) + cls._integer_types):
            orders = list(orders)

        # Forgive if boolean True/False og string 'd'/'c' was passed
        if isinstance(discrete, (Matrix, bool, str)):
            discrete = list(discrete)

        # From here on out, assume `orders` and `discrete` are lists
        if len(orders) < 0:
            raise ValueError('List of orders must have length >=0.')

        if any(i < 0 for i in orders):
            raise ValueError('Every order must be >= 0.')

        if discrete is None:
            discrete = [True] * len(orders)

        if len(orders) != len(discrete):
            raise ValueError('Orders and list of discrete must match length.')

        for p, d in zip(orders, discrete):
            if (not d) and (p not in [0, 1]):
                raise ValueError('Continuous groups must have order 0 or 1.')

        def map_to_bool(x):
            try:
                return cls._repr_dict_inv[x.lower()]
            except:
                return x

        discrete = [map_to_bool(i) for i in discrete]

        return orders, discrete


    def __add__(self, other):
        """
        Override the addition (`+`) operator,
        see :py:meth:`~abelian.groups.LCA.sum`.
        """
        return self.sum(other)

    def __call__(self, element):
        """
        Override function calls,
        see :py:meth:`~abelian.groups.LCA.project_element`.
        """
        return self.project_element(element)

    def __contains__(self, other):
        """
        Override the 'in' operator,
        see :py:meth:`~abelian.groups.LCA.contained_in`.
        """
        return other.contained_in(self)

    def __eq__(self, other):
        """
        Override the equality (`==`) operator,
        see :py:meth:`~abelian.groups.LCA.equal`.
        """
        return self.equal(other)

    def __getitem__(self, key):
        """
        Override the slice operator,
        see :py:meth:`~abelian.groups.LCA.slice`.
        """
        return self.getitem(key)

    def __iter__(self):
        """
        Override the iteration protocol,
        see :py:meth:`~abelian.groups.LCA.iterate`.
        """
        yield from self.iterate()

    def __len__(self):
        """
        Override the ``len()`` function,
        see :py:meth:`~abelian.groups.LCA.length`.
        """
        return self.length()

    def __pow__(self, power, modulo=None):
        """
        Override the pow (`**`) operator,
        see :py:meth:`~abelian.groups.LCA.compose_self`.
        """
        return self.compose_self(power)

    def __repr__(self):
        """
        Override the ``repr()`` function.

        This method is called for computer representation, and if no
        __str__ method is defined this function will be called when ``print()``
        is called on an instance too.
        """

        def repr_single(p, d):
            if p == 0:
                if d:
                    return r'Z'
                return r'R'
            if d:
                return r'Z_' + str(p) + ''
            return r'T' # r'T_' + str(p) + ''

        tuples = self._iterate_tuples()
        joined = r', '.join(repr_single(p, d) for (p, d) in tuples)
        return '[' + joined + ']'

    def canonical(self):
        """
        Return the LCA in canonical form using SNF.

        The canonical form decomposition will:

        (1) Put the torsion (discrete with order >= 1) subgroup in
            a canonical form using invariant factor decomposition from
            the Smith Normal Form decomposition.
        (2) Sort the non-torsion subgroup.

        Returns
        --------
        group : LCA
            The LCA in canonical form.

        Examples
        ---------
        >>> G = LCA([4, 3])
        >>> G.canonical() == LCA([12])
        True
        >>> G = LCA([1, 1, 8, 2, 4], ['c', 'd', 'd', 'd', 'd'])
        >>> G.canonical() == LCA([1], ['c']) + LCA([2, 4, 8])
        True
        """

        def is_t(order, discrete):
            """
            Function to determine if a group is torsion or not.
            """
            if order > 0 and discrete:
                return True
            return False

        def split_torsion(LCA):
            """
            Split a group into (torsion, non_torsion).
            """
            gen_list = list(enumerate(LCA._iterate_tuples()))
            torsion_i = [i for (i, (p, d)) in gen_list if is_t(p, d)]
            nontorsion_i = [i for (i, (p, d)) in gen_list if not is_t(p, d)]
            torsion = LCA.remove_indices(nontorsion_i)
            non_torsion = LCA.remove_indices(torsion_i)
            return torsion, non_torsion

        # Get information pertaining to self
        self_tors, self_non_tors = split_torsion(self)

        # Sort the non-torsion subgroup
        tuples = list(self_non_tors._iterate_tuples())
        non_tors_orders = [p for (p, d) in sorted(tuples)]
        non_tors_discrete = [d for (p, d) in sorted(tuples)]

        # Get canonical decomposition of the torsion subgroup
        self_SNF = smith_normal_form(diag(*self_tors.orders), False)
        self_SNF_p = [p for p in nonzero_diag_as_list(self_SNF) if p != 1]

        # Construct the new group
        orders = non_tors_orders + self_SNF_p
        discrete = non_tors_discrete + [True] * len(self_SNF_p)
        return type(self)(orders = orders, discrete = discrete)

    def copy(self):
        """
        Return a copy of the LCA.

        Returns
        --------
        group : LCA
            A copy of the LCA.

        Examples
        ---------
        >>> G = LCA([1, 5, 7], [False, True, True])
        >>> H = G.copy()
        >>> G == H
        True
        """
        orders_cp = self.orders.copy()
        discrete_cp = self.discrete.copy()
        return type(self)(orders = orders_cp, discrete = discrete_cp)

    def dual(self):
        """
        Return the Pontryagin dual of the LCA.

        Returns a group isomorphic to the Pontryagin dual.

        Returns
        -------
        group : LCA
            The Pontryagin dual of the LCA.

        Examples
        ---------
        >>> G = LCA([5, 1], [True, False])
        >>> H = LCA([5, 0], [True, True])
        >>> G.dual() == H
        True
        >>> G.dual().dual() == G
        True
        >>> self_dual = LCA([5])
        >>> self_dual.dual() == self_dual
        True
        """

        single_dual = self._dual_of_single_group
        dual_lists = list(single_dual(p, d) for (p, d) in self._iterate_tuples())
        new_orders = [p for (p, d) in dual_lists]
        new_discrete = [d for (p, d) in dual_lists]
        return type(self)(orders = new_orders, discrete = new_discrete)

    def elements_by_maxnorm(self, norm_values = None):
        """
        Yield elements corresponding to max norm value.

        If the group is discrete, elements can be generated by maxnorm.

        Parameters
        ----------
        norm_values : iterable
            An iterable containing integer norm values.

        Yields
        -------
        group_element : list
            Group elements with max norm specified by the input iterable.

        Examples
        ---------
        >>> G = LCA([0, 0])
        >>> for element in G.elements_by_maxnorm([0, 1]):
        ...     print(element)
        [0, 0]
        [1, -1]
        [-1, -1]
        [1, 0]
        [-1, 0]
        [1, 1]
        [-1, 1]
        [0, 1]
        [0, -1]
        >>> G = LCA([5, 8])
        >>> for element in G.elements_by_maxnorm([4, 5]):
        ...     print(element)
        [3, 4]
        [4, 4]
        [0, 4]
        [1, 4]
        [2, 4]
        """
        if not all(self.discrete):
            raise TypeError('The group must be discrete.')


        # If the argument is none, create a counter
        if not norm_values:
            norm_values = itertools.count(start = 0, step = 1)
        if isinstance(norm_values, int):
            norm_values = [norm_values]

        orders = self.orders
        for maxnorm_value in norm_values:

            # Keep track of whether ot not a value is yielded
            yielded = False
            for element in elements_of_maxnorm_FGA(orders, maxnorm_value):
                yielded = True
                yield list(element)

            # If no value is yielded, break the possibly infinite loop
            if not yielded:
                break

    def equal(self, other):
        """
        Whether or not two LCAs are equal.

        Two LCAs are equal iff the list of `orders` and the list of `discrete`
        are both equal.

        Parameters
        ----------
        other : LCA
            The LCA to compare equality with.

        Returns
        --------
        equal : bool
            Whether or not the LCAs are equal.

        Examples
        ---------
        >>> G = LCA([1, 5, 7], [False, True, True])
        >>> H = G.copy()
        >>> G == H  # The `==` operator is overloaded
        True
        >>> G.equal(H)  # Equality using the method
        True
        """
        orders_equal = self.orders == other.orders
        discrete_equal = self.discrete ==  other.discrete
        return (orders_equal and discrete_equal)


    def getitem(self, key):
        """
        Return a slice of the LCA.

        Parameters
        ----------
        key: :py:class:`~python.slice`
            A slice object, or an integer.

        Returns
        -------
        group: LCA
            A slice of the FGA as specified by the slice object.

        Examples
        ---------
        >>> G = LCA([5, 6, 1])
        >>> G[0:2] == LCA([5, 6])
        True
        >>> G[0] == LCA([5])
        True
        """

        orders = self.orders[key]
        discrete = self.discrete[key]
        orders = orders if isinstance(orders, list) else [orders]
        discrete = discrete if isinstance(discrete, list) else [discrete]
        return type(self)(orders = orders, discrete = discrete)


    def is_FGA(self):
        """
        Whether or not the LCA is a FGA.

        A locally compact abelian group (LCA) is a finitely generated
        abelian group (FGA) iff all the groups in the direct sum are discrete.

        Returns
        -------
        is_FGA : bool
            True if the object is an FGA, False if not.

        Examples
        ---------
        >>> G = LCA([5, 1], [True, False])
        >>> G.is_FGA()
        False
        >>> G = LCA([1, 7], [False, True])
        >>> G.dual().is_FGA()
        True
        """
        is_FGA = all(self.discrete)
        return is_FGA

    def isomorphic(self, other):
        """
        Whether or not two LCAs are isomorphic.

        Two LCAs are isomorphic iff they can be put into the same
        canonical form.

        Parameters
        ----------
        other : LCA
            The LCA to compare with.

        Returns
        --------
        isomorphic : bool
            Whether or not the LCAs are isomorphic.

        Examples
        ---------
        >>> G = LCA([3, 4])
        >>> H = LCA([12])
        >>> G.isomorphic(H)
        True
        >>> G.equal(H)
        False

        >>> LCA([2, 6]).isomorphic(LCA([12]))
        False

        >>> G = LCA([0, 0, 1, 3, 4], [False, True, True, True, True])
        >>> H = LCA([0, 0, 3, 4], [True, False, True, True])
        >>> G.isomorphic(H)
        True

        >>> LCA([]).isomorphic(LCA.trivial())
        True
        """
        isomorphic = self.canonical().equal(other.canonical())
        return isomorphic

    def iterate(self):
        """
        Yields the groups in the direct sum one by one.

        Iterate through the groups and yield the individual groups in the
        direct sum, one by one.

        Yields
        -------
        group : LCA
            A single group in the direct sum.

        Examples
        ---------
        >>> G = LCA([5, 1], [True, False])
        >>> groups = [LCA([5], [True]), LCA([1], [False])]
        >>> for i, group in enumerate(G):
        ...     group == groups[i]
        True
        True
        """
        for (order, discrete) in zip(self.orders, self.discrete):
            yield type(self)([order], [discrete])

    def length(self):
        """
        The number of groups in the direct sum.

        Returns
        -------
        length : int
            The number of groups in the direct sum.

        Examples
        --------
        >>> G = LCA([])
        >>> G.length()
        0

        >>> G = LCA([0, 1, 1, 5])
        >>> G.length()
        4
        """
        return len(self.orders)

    def compose_self(self, power):
        """
        Repeated direct summation.

        Returns
        -------
        group : LCA
            A new group.

        Examples
        --------
        >>> R = LCA([0], [False])
        >>> (R + R) == R**2
        True
        >>> R**0 == LCA.trivial()
        True
        >>> Z = LCA([0])
        >>> (Z + R)**2 == Z + R + Z + R
        True

        """
        if not isinstance(power, int):
            raise ValueError('Power must be an integer.')
        if power <= 0:
            return self.trivial()
        if power == 1:
            return self
        if power > 1:
            direct_sum = self.copy()
            for prod in range(power - 1):
                direct_sum = direct_sum.sum(self)
            return direct_sum

    def project_element(self, element):
        """
        Project an element onto the group.

        Parameters
        -----------
        element : :py:class:`~sympy.matrices.dense.MutableDenseMatrix` or list
            The group element to project to the LCA.

        Returns
        -------
        element : :py:class:`~sympy.matrices.dense.MutableDenseMatrix` or list
            The group element projected to the LCA.

        Examples
        ---------
        >>> from sympy import Matrix
        >>> G = LCA([5, 9])
        >>> g = [6, 22]
        >>> G.project_element(g)
        [1, 4]
        >>> g = Matrix([13, 13])
        >>> G.project_element(g) == Matrix([3, 4])
        True
        """
        if len(element) in [0, 1] and self.isomorphic(self.trivial()):
            return [0]

        if not (len(element) == len(self.orders) == len(self.discrete)):
            raise ValueError('Length of element must match groups.')

        def project(element, order, discrete):
            if not discrete:
                return mod(element, order)
            if discrete:
                #if mod(element, 1) == 0:
                if isinstance(element, self._integer_types):
                    return mod(element, order)

                raise ValueError('Non-integer cannot be projected to '
                                 'discrete group.')

        generator = zip(element, self.orders, self.discrete)
        projected = [project(e, p, d) for (e, p, d) in generator]

        # If the input is a list, return a list
        if isinstance(element, list):
            return projected

        # If the input is a sympy Matrix, return a sympy Matrix
        if isinstance(element, Matrix):
            return Matrix(projected)


    def rank(self):
        """
        Return the rank of the LCA.

        Returns
        --------
        rank : int
            An integer greater than or equal to 0.

        Examples
        ---------
        >>> G = LCA([5, 6, 1])
        >>> G.rank()
        2

        >>> LCA([1]).rank()
        0

        >>> G = LCA([5, 6, 1])
        >>> H = LCA([1])
        >>> G.rank() + H.rank() == (G + H).rank()
        True
        """

        removed_trivial = self.remove_trivial()
        return removed_trivial.length()

    def remove_indices(self, indices):
        """
        Return a LCA with some groups removed.

        Parameters
        ----------
        indices : list
            A list of indices corresponding to LCAs to remove.

        Returns
        -------
        group : LCA
            The LCA with some groups removed.

        Examples
        --------
        >>> G = LCA([5, 8, 9])
        >>> G.remove_indices([0, 2]) == LCA([8])
        True

        """
        enu = enumerate
        orders = [p for (i, p) in enu(self.orders) if i not in indices]
        discrete = [p for (i, p) in enu(self.discrete) if i not in indices]
        return type(self)(orders = orders, discrete = discrete)

    def remove_trivial(self):
        """
        Remove trivial groups from the object.

        Returns
        --------
        group : LCA
            The group with trivial groups removed.

        Examples
        ---------
        >>> G = LCA([5, 1, 1])
        >>> G.remove_trivial() == LCA([5])
        True
        """
        def trivial(order, discrete):
            return discrete and (order == 1)

        self_tuples = self._iterate_tuples()
        purged_lists = [(p, d) for (p, d) in self_tuples if not trivial(p, d)]
        new_orders = [p for (p, d) in purged_lists]
        new_discrete = [d for (p, d) in purged_lists]
        return type(self)(orders=new_orders, discrete=new_discrete)

    def contained_in(self, other):
        """
        Whether the LCA is contained in `other`.

        A LCA G is contained in another LCA H iff there exists an injection
        from the elements of G to H such that every source/target of the
        mapping is isomorphic. In other words, every group in G must be found
        in H, and no two groups in G can be identified with the same isomorphic
        group is H.

        Parameters
        ----------
        other : LCA
            A locally compact abelian group.

        Returns
        -------
        is_subgroup : bool
            Whether or not `self` is contained in other.

        Examples
        --------
        >>> # Simple example
        >>> G = LCA([2, 2, 3])
        >>> H = LCA([2, 2, 3, 3])
        >>> G.contained_in(H)
        True
        >>> # Order does not matter
        >>> G = LCA([2, 3, 2])
        >>> H = LCA([2, 2, 3, 3])
        >>> G.contained_in(H)
        True
        >>> # Trivial groups are not removed
        >>> G = LCA([2, 3, 2, 1])
        >>> H = LCA([2, 2, 3, 3])
        >>> G in H
        False

        """
        if not isinstance(other, type(self)):
            return TypeError('Must be LCAs.')

        self_as_list = list(self._iterate_tuples())
        other_as_list = list(other._iterate_tuples())

        for group in self_as_list:
            # If the group is not found, return False
            if group not in other_as_list:
                return False

            # If the group is found, remove it
            other_as_list.remove(group)

        # Every group in G is found in H, and it's a subgroup
        return True

    def sum(self, other):
        """
        Return the direct sum of two LCAs.

        Parameters
        ----------
        other : LCA
            The LCA to take direct sum with.

        Returns
        --------
        group : LCA
            The direct sum of self and other.


        Examples
        ---------
        >>> G = LCA([5])
        >>> H = LCA([7])
        >>> G + H == LCA([5, 7])  # The `+` operator is overloaded
        True
        >>> G + H == G.sum(H)  # Directs sums two ways
        True
        """
        new_orders = self.orders + other.orders
        new_discrete = self.discrete + other.discrete
        return type(self)(orders=new_orders, discrete=new_discrete)

    def to_latex(self):
        """
        Return the LCA as a :math:`\LaTeX` string.

        Returns
        -------
        latex_str : str
            A string with LaTeX code for the object.

        Examples
        ---------
        >>> G = LCA([5, 0], [True, False])
        >>> G.to_latex()
        '\\\mathbb{Z}_{5} \\\oplus \\\mathbb{R}'
        """
        def repr_single(p, d):
            if p == 0:
                if d:
                    return r'\mathbb{Z}'
                return r'\mathbb{R}'
            if d:
                return r'\mathbb{Z}_{' + str(p) + '}'
            return r'T' #r'T_{' + str(p) + '}'

        tuples = self._iterate_tuples()
        return r' \oplus '.join(repr_single(p, d) for (p, d) in tuples)

    def _iterate_tuples(self):
        """
        Yields tuples in the form (`order`, `discrete`).
        """
        for group in self:
            yield (group.orders[0], group.discrete[0])

    @staticmethod
    def _dual_of_single_group(order, discrete):
        """
        Compute the dual of a single group.
        """
        if discrete:
        # Discrete
            if order == 0:
                # Dual of Z is T
                return [1, False]  # Return T
            else:
                # Dual of Z_n is Z_n
                return [order, True]  # Return Z_n

        else:
        # Continuous
            if order == 0:
                # Dual of R is R
                return [0, False]  # Return R
            else:
                # Dual of T is Z
                return [0, True]  # Return Z



if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = False)

