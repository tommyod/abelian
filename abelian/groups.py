#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sympy import Matrix
from abelian.utils import mod

class LCA(object):
    """
    Class for Locally Compact Abelian (LCA) groups.
    """


    _repr_dict = {True: 'd', False :'c'}
    _repr_dict_inv = {'d':True, 'c':False}

    def __init__(self, periods, discrete = None):
        """
        Initialize a new Locally Compact Abelian (LCA) group.

        Parameters
        ----------
        periods : list
            A list of periods, e.g. [6, 8, 11].
        discrete : list
            A list of booleans or ['d', 'c', ...] where 'd' stands for
            discrete and 'c' stands for continuous. If None, it defaults to
            discrete.


        Examples
        ---------
        >>> G1 = LCA([5, 6, 7])
        >>> G2 = LCA([5, 6, 7], [True, True, True])
        >>> G3 = LCA([5, 6, 7], ['d']*3)
        >>> (G1 == G2 == G3)
        True
        """

        periods, discrete = self._verify_init(periods, discrete)
        self.periods = periods
        self.discrete = discrete


    @classmethod
    def _verify_init(cls, periods, discrete):
        """
        Verify the user inputs.
        """

        if isinstance(periods, Matrix):
            periods = list(periods)

        if isinstance(discrete, Matrix):
            discrete = list(discrete)

        if len(periods) <= 0:
            raise ValueError('List of periods must have length >0.')

        if any(i < 0 for i in periods):
            raise ValueError('Every period must be >= 0.')

        if discrete is None:
            discrete = [True] * len(periods)

        if len(periods) != len(discrete):
            raise ValueError('Periods and list of discrete must match length.')

        for p, d in zip(periods, discrete):
            if not d and p not in [0, 1]:
                raise ValueError('Continuous groups must have period 0 or 1.')

        def map_to_bool(x):
            try:
                return cls._repr_dict_inv[x.lower()]
            except:
                return x

        discrete = [map_to_bool(i) for i in discrete]

        return periods, discrete


    def is_FGA(self):
        """
        Whether or not the LCA is a FGA.

        Returns
        -------
        order : bool
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
        return all(self.discrete)



    def __repr__(self):
        """
        Representation.
        """

        def repr_single(p, d):
            if p == 0:
                if d:
                    return r'Z'
                return r'R'
            if d:
                return r'Z_' + str(p) + ''
            return r'T_' + str(p) + ''

        joined = r', '.join(repr_single(p, d) for (p, d) in self._gen())
        return '[' + joined + ']'

        repr_dict = self._repr_dict
        return '[' + ', '.join([str(num) + '({})'.format(repr_dict[i]) for (
            num, i) in self._gen()]) + ']'

    def to_latex(self):
        """
        Write object to latex string.
        """
        def repr_single(p, d):
            if p == 0:
                if d:
                    return r'\mathbb{Z}'
                return r'\mathbb{R}'
            if d:
                return r'\mathbb{Z}_{' + str(p) + '}'
            return r'T_{' + str(p) + '}'

        return r' \oplus '.join(repr_single(p, d) for (p, d) in self._gen())

    def dual(self):
        """
        Return the Pontryagin dual.

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
        dual_lists = list(single_dual(p, d) for (p, d) in self._gen())
        new_periods = [p for (p, d) in dual_lists]
        new_discrete = [d for (p, d) in dual_lists]
        return type(self)(periods = new_periods, discrete = new_discrete)

    def _gen(self):
        """
        Yield pairs of (period, discrete).
        """
        for period, discrete in zip(self.periods, self.discrete):
            yield (period, discrete)

    def project_element(self, element):
        """
        Project an element onto the group.

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
        generator = zip(element, self.periods)
        projected = [mod(element, period) for (element, period) in generator]

        # If the input is a list, return a list
        if isinstance(element, list):
            return projected

        # If the input is a sympy Matrix, return a sympy Matrix
        if isinstance(element, Matrix):
            return Matrix(projected)

    def copy(self):
        """
        Return a copy of the LCA.

        Returns
        --------
        group : LCA
            A copy of the object.


        Examples
        ---------
        >>> G = LCA([1, 5, 7], [False, True, True])
        >>> H = G.copy()
        >>> G == H
        True
        """
        return type(self)(periods = self.periods.copy(),
                          discrete = self.discrete.copy())

    def equal(self, other):
        """
        Whether or not two LCAs are equal.

        Parameters
        ----------
        other : LCA
            The LCA to compare with.

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
        return ((self.periods == other.periods) and
                (self.discrete ==  other.discrete))

    def __eq__(self, other):
        """
        Override the equality (`==`) operator.
        """
        return self.equal(other)

    def isomorphic(self):
        """
        TODO.
        """
        pass

    def direct_sum(self, other):
        """
        Returns the direct sum of two LCAs.

        Parameters
        ----------
        other : LCA
            The LCA to take direct sum with.

        Returns
        --------
        group : LCA
            The direct product of self and other.


        Examples
        ---------
        >>> G = LCA([5])
        >>> H = LCA([7])
        >>> G + H == LCA([5, 7])  # The `+` operator is overloaded
        True
        >>> G + H == G.direct_sum(H)  # Directs sums two ways
        True
        """
        new_periods = self.periods + other.periods
        new_discrete = self.discrete + other.discrete
        return type(self)(periods=new_periods, discrete=new_discrete)

    def __add__(self, other):
        """
        Override the addition (`+`) operator.
        """
        return self.direct_sum(other)

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
        def trivial(period, discrete):
            return discrete and (period == 1)

        purged_lists = [(p, d) for (p, d) in self._gen() if not trivial(p, d)]
        new_periods = [p for (p, d) in purged_lists]
        new_discrete = [d for (p, d) in purged_lists]
        return type(self)(periods=new_periods, discrete=new_discrete)

    def rank(self):
        """
        Return the rank of the LCA.

        Returns
        --------
        rank : int
            An integer greater than or equal to 1.

        Examples
        ---------
        >>> G = LCA([5, 6, 1])
        >>> G.rank()
        3
        """
        # TODO: Should trivial groups be removed first?
        return len(self.periods)

    def delete_by_index(self, indices):
        """
        Return a copy with some deleted.

        Parameters
        ----------
        indices

        Returns
        -------

        """
        enu = enumerate
        periods = [p for (i, p) in enu(self.periods) if i not in indices]
        discrete = [p for (i, p) in enu(self.discrete) if i not in indices]
        return type(self)(periods = periods, discrete = discrete)



    @staticmethod
    def _dual_of_single_group(period, discrete):
        """
        Compute the dual of a single group.
        """
        if discrete:
        # Discrete
            if period == 0:
                # Dual of Z is T
                return [1, False]  # Return T
            else:
                # Dual of Z_n is Z_n
                return [period, True]  # Return Z_n

        else:
        # Continuous
            if period == 0:
                # Dual of R is R
                return [0, False]  # Return R
            else:
                # Dual of T is Z
                return [0, True]  # Return Z

    def get_groups(self, slice):
        """
        Return a slice of the LCA.

        Parameters
        ----------
        slice: slice
            A slice object.

        Returns
        -------
        group: LCA
            A slice as specified by the slice object.

        Examples
        ---------
        >>> G = LCA([5, 6, 1])
        >>> G[0:2] == LCA([5, 6])
        True
        >>> G[0] == LCA([5])
        True
        """

        periods = self.periods[slice]
        discrete = self.discrete[slice]
        periods = periods if isinstance(periods, list) else [periods]
        discrete = discrete if isinstance(discrete, list) else [discrete]
        return type(self)(periods = periods, discrete = discrete)

    def __getitem__(self, slice):
        """
        Override the slice (`obj[a:b]`) operator.
        """
        return self.get_groups(slice)

    def __iter__(self):
        """

        Returns
        -------

        """
        if self.is_FGA():
            for period in self.periods:
                yield period

    def __len__(self):
        """
        The rank.

        Returns
        -------

        """
        return len(self.periods)

    def to_list(self):
        """
        Returns periods if FGA.

        Returns
        -------

        """
        if self.is_FGA():
            return self.periods



if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = True)


if __name__ == '__main__':
    G = LCA([0 ,1 ,5 ,0], [True, False, True, False])
    print(G)
    print(G.to_latex())
    print(G.dual())
    print(G.dual().to_latex())
    print(G.dual().remove_trivial())
    print(G.direct_sum(G.dual()))
    print(G.project_element([5, 17, 7, 8.4]))




