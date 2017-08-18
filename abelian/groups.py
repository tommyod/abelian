#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sympy import Matrix, Integer, diag
from abelian.utils import mod
from abelian.linalg.factorizations import smith_normal_form
from abelian.linalg.utils import nonzero_diag_as_list

class LCA:
    """
    A locally compact Abelian group (LCA).
    """

    _repr_dict = {True: 'd', False :'c'}
    _repr_dict_inv = {'d':True, 'c':False}

    _integer_types = (int, Integer)

    def __init__(self, periods, discrete = None):
        """
        Initialize a new LCA.

        Parameters
        ----------
        periods : list
            A list of periods, e.g. [6, 8, 11].
        discrete : list
            A list of booleans or ['d', 'c', ...] where 'd' stands for
            discrete and 'c' stands for continuous. If None, it defaults to
            discrete.


        Examples
        ---------'
        >>> # Create G = Z_5 + Z_6 + Z_7 in three ways
        >>> G1 = LCA([5, 6, 7])
        >>> G2 = LCA([5, 6, 7], [True, True, True])
        >>> G3 = LCA([5, 6, 7], ['d']*3)
        >>> (G1 == G2 == G3)
        True

        >>> # Create G = R + Z
        >>> G = LCA(periods = [0, 0], discrete = [False, True])
        """

        periods, discrete = self._verify_init(periods, discrete)
        self.periods = periods
        self.discrete = discrete


    @classmethod
    def _verify_init(cls, periods, discrete):
        """
        Verify the user inputs.

        Return `periods` and `discrete` as lists.
        """

        if isinstance(periods, Matrix):
            periods = list(periods)

        if isinstance(discrete, Matrix):
            discrete = list(discrete)

        if len(periods) < 0:
            raise ValueError('List of periods must have length >=0.')

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

        A locally compact Abelian group (LCA) is a finitely generated
        Abelian group (FGA) iff all the groups in the direct sum are discrete.

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
            return r'T_{' + str(p) + '}'

        tuples = self._iterate_tuples()
        return r' \oplus '.join(repr_single(p, d) for (p, d) in tuples)

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

        # TODO : Is only allowing T, and not T_{n}, sensible?

        single_dual = self._dual_of_single_group
        dual_lists = list(single_dual(p, d) for (p, d) in self._iterate_tuples())
        new_periods = [p for (p, d) in dual_lists]
        new_discrete = [d for (p, d) in dual_lists]
        return type(self)(periods = new_periods, discrete = new_discrete)

    def _iterate_tuples(self):
        """

        Returns
        -------

        """
        for (period, discrete) in zip(self.periods, self.discrete):
            yield (period, discrete)

    def iterate(self):
        """
        Yields tuples with (`period`, `discrete`).

        Iterate through the `period` and `discrete` lists and yield tuples.

        Yields
        -------
        period : int
            The period of a group.
        discrete : bool
            Whether or not the group is discrete.

        Examples
        ---------
        >>> G = LCA([5, 1], [True, False])
        >>> groups = [LCA([5], [True]), LCA([1], [False])]
        >>> for i, group in enumerate(G):
        ...     group == groups[i]
        True
        True
        """
        for (period, discrete) in zip(self.periods, self.discrete):
            yield type(self)([period], [discrete])

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

        if not (len(element) == len(self.periods) == len(self.discrete)):
            raise ValueError('Length of element must match groups.')

        def project(element, period, discrete):
            if not discrete:
                return mod(element, period)
            if discrete:
                if isinstance(element, self._integer_types):
                    return mod(element, period)

                raise ValueError('Non-integer cannot be projected to '
                                 'discrete group.')

        generator = zip(element, self.periods, self.discrete)
        projected = [project(e, p, d) for (e, p, d) in generator]

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
            A copy of the LCA.

        Examples
        ---------
        >>> G = LCA([1, 5, 7], [False, True, True])
        >>> H = G.copy()
        >>> G == H
        True
        """
        periods_cp = self.periods.copy()
        discrete_cp = self.discrete.copy()
        return type(self)(periods = periods_cp, discrete = discrete_cp)

    def equal(self, other):
        """
        Whether or not two LCAs are equal.

        Two LCAs are equal iff the list of `periods` and the list of `discrete`
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
        periods_equal = self.periods == other.periods
        discrete_equal = self.discrete ==  other.discrete
        return periods_equal and discrete_equal

    def __eq__(self, other):
        """
        Override the equality (`==`) operator,
        see :py:meth:`~abelian.groups.LCA.equal`.
        """
        return self.equal(other)


    def canonical(self):
        """
        Return the LCA in canonical form using SNF.

        The canonical form decomposition will:
        (1) Put the torsion (discrete with period >= 1) subgroup in
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

        def is_t(period, discrete):
            """
            Function to determine if a group is torsion or not.
            """
            if period > 0 and discrete:
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
        non_tors_periods = [p for (p, d) in sorted(tuples)]
        non_tors_discrete = [d for (p, d) in sorted(tuples)]

        # Get canonical decomposition of the torsion subgroup
        self_SNF = smith_normal_form(diag(*self_tors.periods), False)
        self_SNF_p = [p for p in nonzero_diag_as_list(self_SNF) if p != 1]

        # Construct the new group
        periods = non_tors_periods + self_SNF_p
        discrete = non_tors_discrete + [True] * len(self_SNF_p)
        return type(self)(periods = periods, discrete = discrete)


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
        """
        isomorphic = self.canonical().equal(other.canonical())
        return isomorphic


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
        new_periods = self.periods + other.periods
        new_discrete = self.discrete + other.discrete
        return type(self)(periods=new_periods, discrete=new_discrete)

    def __add__(self, other):
        """
        Override the addition (`+`) operator,
        see :py:meth:`~abelian.groups.LCA.sum`.
        """
        return self.sum(other)

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

        self_tuples = self._iterate_tuples()
        purged_lists = [(p, d) for (p, d) in self_tuples if not trivial(p, d)]
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
        1
        """
        # TODO: Is this the correct implementation of RANK?
        # TODO: Or should rank = length?

        return len(self.canonical())

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

        periods = self.periods[key]
        discrete = self.discrete[key]
        periods = periods if isinstance(periods, list) else [periods]
        discrete = discrete if isinstance(discrete, list) else [discrete]
        return type(self)(periods = periods, discrete = discrete)

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
        return len(self.periods)

    def __len__(self):
        """
        Override the ``len()`` function,
        see :py:meth:`~abelian.groups.LCA.length`.
        """
        return self.length()



if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = True)


if __name__ == '__main__':
    G = LCA([0, 0, 0], [False, True, False])
    H = LCA([0, 0, 0], [False, False, True])
    assert G.isomorphic(H)

    G = LCA([0, 0, 0, 1, 1, 3, 1, 4], [False, True, False] + [True] * 5)
    H = LCA([0, 0, 0, 1, 12, 1, 1, 1], [False, False, True] + [True] * 5)
    assert G.isomorphic(H)

    G = LCA([0, 0, 0, 1, 1, 3, 1, 4], [False, True, False] + [True]*5)
    H = LCA([0, 0, 0, 1, 12, 1, 1], [False, False, True] + [True]*4)
    assert G.isomorphic(H)

    G = LCA([], [])
    H = LCA([1])
    assert G.isomorphic(H)

    G = LCA([])
    G = G.canonical()
    print(G.rank())





