#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This module consists of classes representing homomorphisms between
elementary LCAs, the HomLCA class.
"""

from collections.abc import Callable
from sympy import Matrix, diag, latex, Integer, gcd
from abelian.groups import LCA
from abelian.linalg.utils import remove_zero_columns, nonzero_diag_as_list, \
    matrix_mod_vector, order_of_vector, remove_cols, remove_rows, \
    columns_as_list, reciprocal_entrywise, diag_times_mat, mat_times_diag
from abelian.linalg.factorizations import smith_normal_form
from abelian.linalg.factorizations_reals import real_image, real_coimage, \
    real_cokernel, real_kernel
from abelian.linalg.solvers import solve_epi
from abelian.functions import LCAFunc


class HomLCA(Callable):
    """
    A homomorphism between elementary LCAs.
    """

    # The types allowed as entries in A
    _A_all_entry_types = (int, float, complex)
    _A_integer_entry_types = (int, Integer)

    def __init__(self, A, target = None, source = None):
        """
        Initialize a homomorphism.

        Parameters
        ----------
        A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix` or list
            A sympy matrix representing the homomorphism. The user may also
            use a list of lists in the form [row1, row2, ...] as input.
        target : LCA or list
            The target of the homomorphism. If None, a discrete target of
            infinite order is used as the default.
        source : LCA or list
            The source of the homomorphism. If None, a discrete source of
            infinite order is used as the default.

        Examples
        ---------
        >>> # If no source/target is given, a free discrete group is assumed
        >>> phi = HomLCA([[1,2],
        ...               [3,4]])
        >>> phi.source.is_FGA() and phi.target.is_FGA()
        True

        >>> # If no source is given, a free discrete group is assumed
        >>> phi = HomLCA([[1,2],
        ...               [3,4]], target = [5, 5])
        >>> phi.source.is_FGA()
        True

        >>> # The homomorphism must be valid
        >>> from abelian import LCA, HomLCA
        >>> T = LCA(orders = [1], discrete = [False])
        >>> R = LCA(orders = [0], discrete = [False])
        >>> phi = HomLCA([1], target = R, source = T)
        Traceback (most recent call last):
            ...
        ValueError: 1: [T] -> [R] is not homomorphism


        """
        A, target, source = self._verify_init(A, target, source)
        assert isinstance(A, Matrix)
        assert isinstance(target, LCA)
        assert isinstance(source, LCA)
        self.A = A
        self.target = target
        self.source = source

        # TODO : Should we project to target automatically?

    @classmethod
    def identity(cls, group):
        """
        Return the identity morphism.

        Examples
        ---------
        >>> from abelian import LCA, HomLCA
        >>> H = LCA([5, 6, 7])
        >>> G = LCA([0, 0])
        >>> phi = HomLCA([[1,2], [3,4], [5,6]], source = G, target = H)
        >>> Id_H = HomLCA.identity(H)
        >>> Id_G = HomLCA.identity(G)
        >>> # Verify the properties of the identity morphism
        >>> Id_H * phi == phi
        True
        >>> phi * Id_G == phi
        True
        """
        m = group.length()
        return cls(Matrix.eye(m), source = group, target = group)

    @staticmethod
    def _is_scalar_homomorphism(scalar, target, source):
        """
        Check if the source, target and scalar value is a homomorphism.

        Parameters
        ----------
        scalar : float
            A scalar value.
        target : LCA
            An LCA of length 1: either R, T, Z or Z_n.
        source : LCA
            An LCA of length 1: either R, T, Z or Z_n.

        Examples
        ---------
        >>> from abelian import LCA
        >>> Z = LCA([0])
        >>> # This defines a homomorphism
        >>> HomLCA._is_scalar_homomorphism(2, Z, Z)
        True
        >>> # This does not
        >>> HomLCA._is_scalar_homomorphism(2.5, Z, Z)
        False
        >>> # Two examples with FGAs with finite order
        >>> Z_n = LCA([3])
        >>> Z_m = LCA([4])
        >>> HomLCA._is_scalar_homomorphism(4, target = Z_m, source = Z_n)
        True
        >>> HomLCA._is_scalar_homomorphism(3, target = Z_m, source = Z_n)
        False
        """

        # Verify that both groups are of length 1
        if len(source) != 1 or len(target) != 1:
            raise ValueError('Groups must have length 1.')

        # Set up groups here for readability later
        R = LCA([0], [False])
        T = LCA([1], [False])
        Z = LCA([0], [True])
        Z_1 = LCA([1], [True])

        # Trivial homomorphism is always valid
        if scalar == 0 or source == Z_1:
            return True

        # Defined for readability
        def integer_value(arg):
            return arg % 1 == 0

        # The source group is R
        if source == R:
            if target == R or target == T:
                return True
            else:
                return False

        # The source group is T
        if source == T:
            if target == T and integer_value(scalar):
                return True
            else:
                return False

        # The source group is Z
        if source == Z:
            if target == R or target == T:
                return True
            if integer_value(scalar):
                return True
            return False

        # The source group is Z_n
        n = source.orders[0]
        if source.discrete[0] is True and n > 0:
            if target == R or target == Z:
                return False

            if target == T and integer_value(scalar * n):
                return True

            m = target.orders[0]
            if target.discrete[0] is True and m > 0:
                if integer_value((scalar * gcd(m, n) / m)):
                    return True
                else:
                    return False

        return False


    @classmethod
    def _verify_init(cls, A, target, source):
        """
        Verify the inputs.

        Parameters
        ----------
        A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix` or list
            A sympy matrix representing the homomorphism. The user may also
            use a list of lists in the form [row1, row2, ...] as input.
        target : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`,
        list or LCA
            The target of the homomorphism. If None, a discrete target of
            infinite order is used as the default.
        source : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`,
        list or LCA
            The source of the homomorphism. If None, a discrete source of
            infinite order is used as the default.

        Returns
        -------
        A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        target : LCA
        source : LCA
        """

        # If the input matrix is a list of lists, convert to matrix
        if isinstance(A, list):
            A = Matrix(A)
        elif isinstance(A, cls._A_all_entry_types):
            A = Matrix([A])


        # If no target is given, assume free-to-free morphism
        if (target is None):
            target = [0] * A.rows
            source = [0] * A.cols

        # If target is given, and no source, assume left-free morphism
        elif (source is None):
            source = [0] * A.cols

        # If lists of matrices were passed, convert to LCAs
        if isinstance(target, (list, Matrix)):
            target = LCA(list(target))

        if isinstance(source, (list, Matrix)):
            source = LCA(list(source))

        # If the matrix is empty on either side, replace the LCA
        m, n = A.shape
        if m == 0:
            return A, LCA([]), source
        if n == 0:
            return A, target, LCA([])



        # Verify that the input is actually a homomorphism
        for j, s in enumerate(source):
            for i, t in enumerate(target):
                if not cls._is_scalar_homomorphism(A[i, j], t, s):
                    data = A[i, j], s, t
                    err = "{}: {} -> {} is not homomorphism".format(*data)
                    raise ValueError(err)


        return A, target, source


    def __add__(self, other):
        """
        Override the addition (`+`) operator,
        see :py:meth:`~abelian.morphisms.HomLCA.add`.
        """
        return self.add(other)


    def __call__(self, source_element):
        """
        Override function calls,
        see :py:meth:`~abelian.morphisms.HomLCA.evaluate`.
        """
        return self.evaluate(source_element)


    def __eq__(self, other):
        """
        Override the equality (`==`) operator,
        see :py:meth:`~abelian.morphisms.HomLCA.equal`.
        """
        return self.equal(other)

    def __getitem__(self, args):
        """
        Override the slice operator,
        see :py:meth:`~abelian.morphisms.HomLCA.getitem`.
        """
        return self.getitem(args)


    def __mul__(self, other):
        """
        Override the `*` operator,
        see :py:meth:`~abelian.morphisms.HomLCA.compose`.
        """
        return self.compose(other)

    def __pow__(self, power, modulo=None):
        """
        Override the pow (`**`) operator,
        see :py:meth:`~abelian.morphisms.HomLCA.compose_self`.
        """
        return self.compose_self(power)

    def __radd__(self, other):
        """
        Override the addition (`+`) operator,
        see :py:meth:`~abelian.morphisms.HomLCA.add`.
        """
        return self.add(other)


    def __repr__(self):
        """
        Override the ``repr()`` function.
        """
        rep = 'source: {}     target: {}\n'.format(repr(self.source),
                                                   repr(self.target))
        rep += str(self.A)
        return rep


    def __rmul__(self, other):
        """
        Override the `*` operator,
        see :py:meth:`~abelian.morphisms.HomLCA.compose`.
        """
        return self.compose(other)

    def add(self, other):
        """
        Elementwise addition.

        Elementwise addition of the underlying matrix.

        Parameters
        ----------
        other : HomLCA or numeric
            A homomorphism to add to the current one, or a number.

        Returns
        -------
        homomorphism : HomLCA
            A new homomorphism with the argument added.

        """
        # If the `other` argument is a numeric type
        if isinstance(other, self._A_all_entry_types):
            m, n = self.A.shape
            new_A = self.A + Matrix(m, n, lambda i,j : other)
            return type(self)(new_A, target=self.target, source=self.source)

        # If the `other` argument is a numeric type
        if isinstance(other, type(self)):
            if not self.target == other.target and self.source == other.source:
                raise ValueError('Sources and targets must match..')

            new_A = self.A + other.A
            return type(self)(new_A, target=self.target, source=self.source)

        format_args = type(self), type(other)
        raise ValueError('Cannot add {} and {}'.format(*format_args))


    def compose(self, other):
        """
        Compose two homomorphisms.

        The composition of `self` and `other` is first other, then self.

        Parameters
        ----------
        other : HomLCA
            The homomorphism to compose with.

        Returns
        -------
        homomorphism : HomLCA
            The composition of `self` and `other`, i.e. `self` ( `other` (x)).

        Examples
        --------
        >>> phi = HomLCA([[1, 0, 1],
        ...               [0, 1, 1]])
        >>> ker_phi = HomLCA([1, 1, -1])
        >>> (phi * ker_phi) == HomLCA([0, 0])
        True
        >>> phi.compose(ker_phi) == HomLCA([0, 0])
        True
        """

        # If the `other` argument is a numeric type
        if isinstance(other, self._A_all_entry_types):
            new_A = self.A * other
            return type(self)(new_A, target=self.target, source=self.source)

        # If the `other` argument is a HomLCA
        if isinstance(other, type(self)):
            if not other.target == self.source:
                raise ValueError('Target of other must equal source of self.')

            new_A = self.A * other.A
            return type(self)(new_A, target=self.target, source=other.source)

        # If the composition is an HomLCA, then a LCAFunc,
        # it's valid if the domain matches
        if isinstance(other, LCAFunc):
            if other.domain.equal(self.target):
                return other.pullback(self)

        format_args = type(self), type(other)
        raise ValueError('Cannot compose/add {} and {}'.format(*format_args))

    def compose_self(self, power):
        """
        Repeated composition of an endomorphism.

        Parameters
        ----------
        power : int
            The number of times to compose with self.

        Returns
        -------
        homomorphism : HomLCA
            The endomorphism composed with itself `power` times.

        Examples
        --------
        >>> from sympy import diag
        >>> phi = HomLCA(diag(2, 3))
        >>> phi**3 == HomLCA(diag(2**3, 3**3))
        True
        """
        product = self.copy()
        for prod in range(power - 1):
            product = product.compose(self)
        return product

    def copy(self):
        """
        Return a copy of the homomorphism.

        Returns
        -------
        homomorphism : HomLCA
            A copy of the homomorphism.

        Examples
        --------
        >>> phi = HomLCA([1, 2, 3])
        >>> phi.copy() == phi
        True
        """
        return type(self)(self.A, target = self.target, source = self.source)

    def dual(self):
        """
        Compute the dual homomorphism.

        TODO: Write detailed description.


        Returns
        -------
        dual : HomLCA
            The dual homomorphism.

        Examples
        ---------
        >>> phi = HomLCA([2])
        >>> phi_dual = phi.dual()
        >>> phi_dual.source == phi_dual.target
        True

        Computing duals by first calculating orders

        >>> # Project, then find dual
        >>> phi = HomLCA([2], target = [10])
        >>> phi_proj = phi.project_to_source()
        >>> phi_project_dual = phi_proj.dual()
        >>> phi_project_dual == HomLCA([1], [5], [10])
        True
        >>> # Do not project
        >>> phi_dual = phi.dual()
        >>> phi_dual == HomLCA([1/5], LCA([1], [False]), [10])
        True
        """

        # Flip the source and target for the dual morphism
        dual_source, dual_target = self.target.dual(), self.source.dual()

        # Calculate the matrix representing the dual homomorphism
        diag_p = Matrix([1 if e == 0 else e for e in self.source.orders])
        diag_q_inv = reciprocal_entrywise(Matrix([1 if e == 0 else e for e
                                                  in self.target.orders]))
        dual_A = mat_times_diag(diag_times_mat(diag_p, self.A.T), diag_q_inv)

        # Create new FGA object and return
        return HomLCA(dual_A, target = dual_target, source = dual_source)

    def equal(self, other):
        """
        Whether or not two homomorphisms are equal.

        Two HomLCAs are equal iff (1) the sources are equal, (2) the targets
        are equal and (3) the matrices representing the homomorphisms are
        equal.

        Parameters
        ----------
        other : HomLCA
            A HomLCA to compare equality with.

        Returns
        -------
        equal : bool
            Whether or not the HomLCAs are equal.

        Examples
        ---------
        >>> phi = HomLCA([1], target=[0], source = [0]) # Explicit
        >>> psi = HomLCA([1])   # Shorter, defaults to the above
        >>> phi == psi
        True

        """
        source_equal = self.source == other.source
        target_equal = self.target == other.target
        A_equal = self.A == other.A
        return source_equal and target_equal and A_equal

    def evaluate(self, source_element):
        """
        Apply the homomorphism to an element.

        Parameters
        ----------
        source_element

        Returns
        -------

        Examples
        ---------
        >>> from sympy import diag
        >>> phi = HomLCA(diag(3, 4), target = [5, 6])
        >>> phi.evaluate([2, 3])
        [1, 0]
        >>> phi.evaluate(Matrix([2, 3]))
        Matrix([
        [1],
        [0]])
        """

        # Project the element to the source LCA of the homomorphism
        source_element = self.source.project_element(source_element)

        m, n = self.A.shape
        if n == 0:
            evaluated = [0]

        # Apply the homomorphism using matrix multiplication, depending
        # on the type of the input
        elif isinstance(source_element, Matrix):
            evaluated = self.A * source_element
        else:
            evaluated = self.A * Matrix(source_element)

        # Project to the target LCA of the homomorphism
        projected = self.target.project_element(evaluated)

        # Return the same type of data as the input
        if isinstance(source_element, Matrix):
            return projected
        else:
            rows = columns_as_list(projected.T)
            return [r[0] for r in rows]

    def getitem(self, args):
        """
        Return a slice of the homomorphism.

        Slices the object with the common matrix slice notation, e.g. A[rows,
        cols], where the `rows` and `cols` objects can be either
        integers or slice objects. If the homomorphism is represented by a
        column or row matrix, then the notation A[key] will also work.
        The underlying matrix and the source and target LCAs are all sliced.

        Parameters
        ----------
        args : slice
            A slice or a tuple with (slice_row, slice_col).

        Returns
        --------
        homomorphism : HomLCA
            A sliced homomorphism.

        Examples
        --------

        The homomorphism is sliced using two input arguments.

        >>> from sympy import diag
        >>> phi = HomLCA(diag(4,5,6))
        >>> phi[0,:] == HomLCA([[4, 0, 0]])
        True
        >>> phi[:,1] == HomLCA([0, 5, 0])
        True

        If the homomorphism is represented by a row or column, one arg will do.

        >>> phi = HomLCA([1,2,3])
        >>> phi[0:2] == HomLCA([1,2])
        True
        """
        # Two arguments passed, slice row and column of A
        if isinstance(args, (slice, int)):
            # One argument was passed
            rows_A, cols_A = self.A.shape
            slice1 = args
            if cols_A == 1:
                # A column vector
                new_A = self.A[slice1]
                new_target = self.target[slice1]
                new_source = self.source
                return type(self)(new_A, target=new_target, source=new_source)
            elif rows_A == 1:
                # A column vector
                new_A = self.A[slice1]
                new_target = self.target
                new_source = self.source[slice1]
                return type(self)(new_A, target=new_target, source=new_source)
            else:
                error = 'One slice argument was passed, but the morphism has' \
                        ' dimension {} x {}'.format(rows_A, cols_A)
                return ValueError(error)


        elif isinstance(args, tuple) and len(args) == 2:

            # Two arguments were passed, slice the matrix and rows/cols
            slice1, slice2 = args
            if not (isinstance(slice1, (slice, int)) and
                isinstance(slice1, (slice, int))):
                raise ValueError('Arguments must be slice() or integer.')

            new_A = self.A[slice1, slice2]
            new_target = self.target[slice1]
            new_source = self.source[slice2]
            return type(self)(new_A, target=new_target, source=new_source)

        else:
            raise ValueError('slice() takes 1 or 2 arguments.')


    def remove_trivial_groups(self):
        """
        Remove trivial groups.

        A group is trivial if it is discrete with order 1, i.e. Z_1.
        Removing trivial groups from the target group means removing the
        Z_1 groups from the target, along with the corresponding rows of
        the matrix representing the homomorphism.
        Removing trivial groups from the source group means removing the
        groups Z_1 from the source, i.e. removing every column (generator)
        with order 1.

        Returns
        -------
        homomorphism : HomLCA
            A homomorphism where the trivial groups have been removed from
            the source and the target. The corresponding rows and columns of
            the matrix representing the homomorphism are also removed.

        Examples
        --------
        >>> target = [1, 7]
        >>> phi = HomLCA([[2, 1], [7, 2]], target=target)
        >>> projected = HomLCA([[2]], target=[7], source = [7])
        >>> phi.project_to_source().remove_trivial_groups() == projected
        True

        """
        def trivial(order, discrete):
            return discrete and (order == 1)

        # Get indices where the value of the source is 1
        generator = enumerate(self.source._iterate_tuples())
        cols_to_del = [i for (i, (d, p)) in generator if trivial(d, p)]
        new_A = remove_cols(self.A, cols_to_del)

        # Get indices where the value of the target is 1
        generator = enumerate(self.target._iterate_tuples())
        rows_to_del = [i for (i, (d, p)) in generator if trivial(d, p)]
        new_A = remove_rows(new_A, rows_to_del)

        new_source = self.source.remove_indices(cols_to_del)
        new_target = self.target.remove_indices(rows_to_del)

        return type(self)(new_A, new_target, new_source)


    def stack_diag(self, other):
        """
        Stack diagonally.

        Parameters
        ----------
        other : HomLCA
            A homomorphism to stack with the current one.

        Returns
        -------
        stacked_vert : HomLCA
            The result of stacking the homomorphisms on diagonally.

        Examples
        --------
        >>> phi = HomLCA([1])
        >>> psi = HomLCA([2])
        >>> phi.stack_diag(psi) == HomLCA([[1, 0], [0, 2]])
        True

        """
        new_source = self.source + other.source
        new_target = self.target + other.target
        new_A = diag(self.A, other.A)

        return type(self)(new_A, target = new_target, source = new_source)

    def stack_horiz(self, other):
        """
        Stack horizontally (column wise).

        The targets must be the same, the sources will be concatenated.
        The stacking is done to create a matrix with structure [self,
        other], i.e. "Putting `self` to the left of `other`."

        Parameters
        ----------
        other : HomLCA
            A homomorphism to stack with the current one.

        Returns
        -------
        stacked_vert : HomLCA
            The result of stacking the homomorphisms side by side.

        Examples
        --------
        >>> phi = HomLCA([1])
        >>> psi = HomLCA([2])
        >>> phi.stack_horiz(psi) == HomLCA([[1, 2]])
        True

        """
        if not self.target == other.target:
            raise ValueError('Targets must be equal to stack horizontally.')
        new_source = self.source + other.source
        new_target = self.target
        new_A = self.A.row_join(other.A)
        return type(self)(new_A, target = new_target, source = new_source)


    def stack_vert(self, other):
        """
        Stack vertically (row wise).

        The sources must be the same, the targets will be concatenated.
        The stacking is done to create a matrix with structure [[self],
        [other]], i.e. "Putting `self` on top of of `other`."

        Parameters
        ----------
        other : HomLCA
            A homomorphism to stack with the current one.

        Returns
        -------
        stacked_vert : HomLCA
            The result of stacking the homomorphisms on top of each other.

        Examples
        --------
        >>> phi = HomLCA([1])
        >>> psi = HomLCA([2])
        >>> phi.stack_vert(psi) == HomLCA([1, 2])
        True

        """
        if not self.source == other.source:
            raise ValueError('Sources must be equal to stack vertically.')
        new_source = self.source
        new_target = self.target + other.target
        new_A = self.A.col_join(other.A)
        return type(self)(new_A, target = new_target, source = new_source)

    def update(self, new_A = None, new_target = None, new_source = None):
        """
        Return a new homomorphism with updated properties.
        """

        if new_A is None:
            new_A = self.A
        if new_target is None:
            new_target = self.target
        if new_source is None:
            new_source = self.source

        return type(self)(new_A, target=new_target, source=new_source)


    def _is_homFGA(self):
        """
        Whether or not is a homomorphism between FGAs.

        Returns
        -------
        homFGA : bool
            Whether or not it's a homFGA.

        Examples
        ---------
        >>> phi = HomLCA([1], source = [1], target = [1])
        >>> phi._is_homFGA()
        True

        """
        integer_entries = all([i % 1 == 0 for i in self.A])
        if self.source.is_FGA() and self.target.is_FGA() and integer_entries:
            return True
        else:
            return False

    def to_latex(self):
        """
        Return the homomorphism as a :math:`\LaTeX` string.

        Returns
        -------
        latex : str
            The HomLCA formatted as a LaTeX string.

        Examples
        --------
        >>> phi = HomLCA([1])
        >>> phi.to_latex()
        '\\\\begin{pmatrix}1\\\\end{pmatrix}:\\\\mathbb{Z} \\\\to \\\\mathbb{Z}'
        """
        latex_code = latex(self.A)
        latex_code = latex_code.replace(r'\left[\begin{matrix}',
                                        r'\begin{pmatrix}')
        latex_code = latex_code.replace(r'\end{matrix}\right]',
                                        r'\end{pmatrix}')
        format_args = self.source.to_latex(), self.target.to_latex()
        latex_code += r':{} \to {}'.format(*format_args)
        return latex_code


    @classmethod
    def zero(cls, target, source):
        """
        Initialize the zero morphism.

        Parameters
        ----------
        target : LCA or list
            The target of the homomorphism. If None, a discrete target of
            infinite order is used as the default.
        source : LCA or list
            The source of the homomorphism. If None, a discrete source of
            infinite order is used as the default.

        Examples
        ---------
        >>> zero = HomLCA.zero([0]*3, [0]*3)
        >>> zero([1, 5, 7]) == [0, 0, 0]
        True
        """
        rows = len(target)
        cols = len(source)
        A = Matrix(rows, cols, lambda i, j : 0)
        return cls(A, target = target, source = source)

    @property
    def shape(self):
        """
        The shape (`rows`, `cols`).

        Returns
        -------
        shape : tuple
            A tuple with the shape of the underlying matrix A, i.e. (`rows`,
            `cols`).

        """
        return self.A.shape


    def annihilator(self):
        """
        Compute the annihilator monomorphism.
        """
        # TODO: Write this method.
        return self.cokernel().dual().remove_trivial_groups()

    def coimage(self):
        """
        Compute the coimage epimorphism.

        Returns
        -------
        homomorphism : HomLCA
            The coimage homomorphism.

        Examples
        --------
        >>> phi = HomLCA([[4, 4],
        ...               [2, 8]], target = [16, 16])
        >>> im = phi.image().remove_trivial_groups()
        >>> coim = phi.coimage().remove_trivial_groups()
        >>> phi == (im * coim).project_to_target()
        True

        """
        R = LCA([0], [False])
        target_real_noncompact = all(g == R for g in self.target)
        A_is_integer = all(isinstance(a, self._A_integer_entry_types) for a
                           in self.A)


        # If the mapping is R^d -> R^d, the coimage is found with the SVD
        if all(g == R for g in self.source) and \
            all(g == R for g in self.target):
            coimage_A = real_coimage(self.A)
            m, n = coimage_A.shape
            return type(self)(coimage_A, source=R**n, target=R**m)

        if self.source.is_FGA() and target_real_noncompact and A_is_integer:
            return self._FGA_coimage()

        if self._is_homFGA():
            return self._FGA_coimage()
        else:
            raise NotImplementedError('Not implemented.')

    def cokernel(self):
        """
        Compute the cokernel epimorphism.

        Returns
        -------
        homomorphism : HomLCA
            The cokernel homomorphism.

        Examples
        --------
        >>> phi = HomLCA([[1, 0], [0, 1], [1, 1]])
        >>> coker = phi.cokernel()
        >>> coker.target.isomorphic(LCA([1, 1, 0]))
        True

        """


        if self._is_homFGA():
            return self._FGA_cokernel()

        # If it's a map Z^m -> R^n, then return inverse matrix
        Z = LCA(orders = [0], discrete = [True])
        R = LCA(orders=[0], discrete=[False])
        T = LCA(orders=[1], discrete=[False])
        m, n = self.shape

        # If the mapping is R^d -> R^d, the kernel is found with the SVD
        if all(g == R for g in self.source) and \
            all(g == R for g in self.target):
            cokernel_A = real_cokernel(self.A)
            m, n = cokernel_A.shape
            return type(self)(cokernel_A, source=R**n, target=R**m)


        if (m == n) and self.source == Z**m and self.target == R**m:
            inverse_A = self.A.inv()
            return type(self)(inverse_A, target = T**m, source = self.target)

        else:
            raise NotImplementedError('Not implemented.')

    def image(self):
        """
        Compute the image monomorphism.

        Returns
        -------
        homomorphism : HomLCA
            The image homomorphism.

        Examples
        --------
        >>> phi = HomLCA([[4, 4],
        ...               [2, 8]], target = [64, 32])
        >>> im = phi.image().remove_trivial_groups()
        >>> coim = phi.coimage().remove_trivial_groups()
        >>> phi == (im * coim).project_to_target()
        True

        >>> # Image computations are also allowed when target is R
        >>> R = LCA(orders = [0], discrete = [False])
        >>> sample_matrix = [[1, 2, 3], [2, 3, 5]]
        >>> phi_sample = HomLCA(sample_matrix, target = R + R)
        >>> phi_sample_im = phi_sample.image().remove_trivial_groups()
        >>> phi_sample_im == phi_sample[:, 1:]
        True

        """
        R = LCA([0], [False])
        target_real_noncompact = all(g == R for g in self.target)
        A_is_integer = all(isinstance(a, self._A_integer_entry_types) for a
                           in self.A)


        # If the mapping is R^d -> R^d, the image is found with the SVD
        if all(g == R for g in self.source) and \
            all(g == R for g in self.target):
            image_A = real_image(self.A)
            m, n = image_A.shape
            return type(self)(image_A, source=R**n, target=R**m)


        if self.source.is_FGA() and target_real_noncompact and A_is_integer:
            return self._FGA_image()
        if self._is_homFGA():
            return self._FGA_image()
        else:
            raise NotImplementedError('Not implemented.')


    def kernel(self):
        """
        Compute the kernel monomorphism.

        Returns
        -------
        homomorphism : HomLCA
            The kernel homomorphism.

        Examples
        --------
        >>> phi = HomLCA([[1, 0, 1], [0, 1, 1]])
        >>> phi.kernel() == HomLCA([-1, -1, 1])
        True

        """
        R = LCA([0], [False])
        T = LCA([1], [False])
        target_real_noncompact = all(g == R for g in self.target)
        A_is_integer = all(isinstance(a, self._A_integer_entry_types) for a
                           in self.A)

        # If the mapping is R^d -> T^d, the kernel is the inverse
        m, n = self.shape
        if m == n:
            if self.source == R**m and self.target == T**m:
                kernel_source = LCA([0], [True])**m
                A_inv = self.A.inv()
                return type(self)(A_inv, source = kernel_source, target =
                self.source)

        # If the mapping is R^d -> R^d, the kernel is found with the SVD
        if all(g == R for g in self.source) and \
            all(g == R for g in self.target):
            kernel_A = real_kernel(self.A)
            m, n = kernel_A.shape
            return type(self)(kernel_A, source=R**n, target=R**m)


        if self.source.is_FGA() and target_real_noncompact and A_is_integer:
            return self._FGA_kernel()

        if self._is_homFGA():
            return self._FGA_kernel()
        else:
            raise NotImplementedError('Not implemented.')

    def _FGA_coimage(self):
        """
        Compute the coimage epimorphism.

        Returns
        -------
        homomorphism : HomLCA
            The coimage homomorphism.

        Examples
        --------
        >>> phi = HomLCA([[4, 4],
        ...               [2, 8]], target = [16, 16])
        >>> im = phi.image().remove_trivial_groups()
        >>> coim = phi.coimage().remove_trivial_groups()
        >>> phi == (im * coim).project_to_target()
        True

        """
        # Compute the cokernel of the kernel and return
        kernel = self.kernel()
        coimage = kernel.cokernel()
        return coimage

    def _FGA_cokernel(self):
        """
        Compute the cokernel epimorphism.

        Returns
        -------
        homomorphism : HomLCA
            The cokernel homomorphism.

        Examples
        --------
        >>> phi = HomLCA([[1, 0], [0, 1], [1, 1]])
        >>> coker = phi.cokernel()
        >>> coker.target.isomorphic(LCA([1, 1, 0]))
        True

        """
        # Horizontally stack A and ker(pi_2)
        A_ker_pi = self.A.row_join(remove_zero_columns(diag(
            *self.target.orders)))
        # Compute SNF, get size and the kernel
        U, S, V = smith_normal_form(A_ker_pi)
        diagonal = nonzero_diag_as_list(S)
        (m, n), r = self.A.shape, len(diagonal)
        quotient = diagonal + [0] * (m - r)

        # Initialize the cokernel morphism and project it onto the target
        coker = type(self)(U, target = quotient, source=self.target)
        return coker.project_to_target()

    def _FGA_image(self):
        """
        Compute the image monomorphism.

        Returns
        -------
        homomorphism : HomLCA
            The image homomorphism.

        Examples
        --------
        >>> phi = HomLCA([[4, 4],
        ...               [2, 8]], target = [64, 32])
        >>> im = phi.image().remove_trivial_groups()
        >>> coim = phi.coimage().remove_trivial_groups()
        >>> phi == (im * coim).project_to_target()
        True

        """
        # Solve equation for the image
        coim = self.coimage()
        coim_target = Matrix(coim.target.orders)
        solved_mat = solve_epi(coim.A, self.A, coim_target)

        # Initialize morphism and return project onto target
        image = type(self)(solved_mat, self.target, source=coim.target)
        return image.project_to_target()

    def _FGA_isomorphic(self, other):
        """
        Whether or not two HomLCAs are isomorphic.

        Two homomorphisms are isomorphic iff they generate the same group.

        Parameters
        ----------
        other : HomLCA
            The homomorphism to compare with.

        Returns
        -------
        isomorphic : bool
            Whether or not `self` and `other` are isomorphic.

        Examples
        --------
        >>> from sympy import diag
        >>> # The order does not matter
        >>> phi = HomLCA(diag(2, 3))
        >>> psi = HomLCA(diag(3, 2))
        >>> phi._FGA_isomorphic(psi)
        True
        >>> # These homomorphisms both generate Z_2 + Z
        >>> phi = HomLCA([[2, 0],
        ...               [0, 1]], target=[4, 0])
        >>> psi = HomLCA([[1, 0],
        ...               [0, 1]], target=[2, 0])
        >>> phi._FGA_isomorphic(psi)
        True
        >>> # They both generate a group isomorphic to Z_6
        >>> phi = HomLCA([[2, 0], [0, 1]], target=[4, 3])
        >>> psi = HomLCA([2, 2], target=[4, 3])
        >>> phi._FGA_isomorphic(psi)
        True
        """
        # TODO: Is this correct?


        self_no_triv = self.remove_trivial_groups().image()
        other_no_triv = other.remove_trivial_groups().image()

        isomorphic = self_no_triv.source.isomorphic(other_no_triv.source)

        return isomorphic

    def _FGA_kernel(self):
        """
        Compute the kernel monomorphism.

        Returns
        -------
        homomorphism : HomLCA
            The kernel homomorphism.

        Examples
        --------
        >>> phi = HomLCA([[1, 0, 1], [0, 1, 1]])
        >>> phi.kernel() == HomLCA([-1, -1, 1])
        True

        """
        # Horizontally stack A and ker(pi_2)
        orders = self.target.orders
        A_ker_pi = self.A.row_join(remove_zero_columns(diag(*orders)))

        # Compute SNF, get size and the kernel
        U, S, V = smith_normal_form(A_ker_pi)
        (m, n), r = self.A.shape, len(nonzero_diag_as_list(S))
        kernel_matrix = V[:n, r:]

        kernel = type(self)(kernel_matrix, target = self.source)
        return kernel.project_to_target()

    def project_to_source(self):
        """
        Project columns to source group (orders).

        Returns
        -------
        homomorphism : HomLCA
            A homomorphism with orders in the source FGA.

        Examples
        --------
        >>> target = [3, 6]
        >>> phi = HomLCA([[1, 0],
        ...               [3, 3]], target = target)
        >>> phi = phi.project_to_source()
        >>> phi.source.orders == [6, 2]
        True
        """
        A_is_integer = all(isinstance(a, self._A_integer_entry_types) for a
                           in self.A)

        if A_is_integer and self.source.is_FGA():
            # Find dimensions
            m, n = self.A.shape

            # Compute orders for all columns of A
            target_vect = Matrix(self.target.orders)
            source = [order_of_vector(self.A[:, i], target_vect) for i in range(n)]
            return type(self)(self.A, self.target, source)
        else:
            raise NotImplementedError('Not implemented: projection to target.')

    def project_to_target(self):
        """
        Project columns to target group.

        Returns
        -------
        homomorphism : HomLCA
            A homomorphism with columns projected to the target FGA.

        Examples
        --------
        >>> target = [7, 12]
        >>> phi = HomLCA([[15, 12],
        ...               [9,  17]], target = target)
        >>> phi_proj = HomLCA([[1, 5],
        ...                    [9, 5]], target = target)
        >>> phi.project_to_target() == phi_proj
        True
        """

        A = matrix_mod_vector(self.A, Matrix(self.target.orders))
        return type(self)(A, target = self.target, source = self.source)

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = False)
