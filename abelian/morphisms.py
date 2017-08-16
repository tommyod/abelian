#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types
from sympy import Matrix, diag, latex
from abelian.utils import mod
from abelian.groups import LCA
from abelian.linalg.utils import delete_zero_columns, nonzero_diag_as_list, \
    matrix_mod_vector, order_of_vector, remove_cols, remove_rows, columns_as_list
from abelian.linalg.factorizations import smith_normal_form
from abelian.linalg.solvers import solve_epi


class HomLCA:
    """
    A homomorphism between LCAs.
    """

    # The types allowed as entries in A
    _A_entry_types = (int, float, complex)

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
            infinite period is used as the default.
        source : LCA or list
            The source of the homomorphism. If None, a discrete source of
            infinite period is used as the default.

        Examples
        ---------
        >>> phi = HomLCA([[1,2],
        ...               [3,4]])
        """

        A, target, source = self._verify_init(A, target, source)
        assert isinstance(target, LCA)
        assert isinstance(source, LCA)
        self.A = A
        self.target = target
        self.source = source

    @staticmethod
    def _verify_init(A, target, source):
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
            infinite period is used as the default.
        source : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`,
        list or LCA
            The source of the homomorphism. If None, a discrete source of
            infinite period is used as the default.

        Returns
        -------
        A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        target : LCA
        source : LCA
        """

        # If the input matrix is a list of lists, convert to matrix
        if isinstance(A, list):
            A = Matrix(A)

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

        return A, target, source


    def to_latex(self):
        """
        Return the HomLCA as a :math:`\LaTeX` string.

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


    def __getitem__(self, args):
        """
        Override the slice operator,
        see :py:meth:`~abelian.morphisms.HomLCA.getitem`.
        """
        return self.getitem(args)

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
        >>> phi = HomFGA(diag(4,5,6))
        >>> phi[0,:] == HomFGA([[4, 0, 0]])
        True
        >>> phi[:,1] == HomFGA([0, 5, 0])
        True

        If the homomorphism is represented by a row or column, one arg will do.

        >>> phi = HomFGA([1,2,3])
        >>> phi[0:2] == HomFGA([1,2])
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
            print(args)
            raise ValueError('slice() takes 1 or 2 arguments.')


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
        if isinstance(other, self._A_entry_types):
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

    def __add__(self, other):
        """
        Override the addition (`+`) operator,
        see :py:meth:`~abelian.morphisms.HomLCA.add`.
        """
        return self.add(other)

    def __radd__(self, other):
        """
        Override the addition (`+`) operator,
        see :py:meth:`~abelian.morphisms.HomLCA.add`.
        """
        return self.add(other)


    def compose_self(self, power):
        """
        Repeated composition of an automorphism.

        Parameters
        ----------
        power : int
            The number of times to compose with self.

        Returns
        -------
        homomorphism : HomLCA
            The automorphism composed with itself `power` times.

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


    def __pow__(self, power, modulo=None):
        """
        Override the pow (`**`) operator,
        see :py:meth:`~abelian.morphisms.HomLCA.compose_self`.
        """
        return self.compose_self(power)

    def copy(self):
        """
        Return a copy.

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
        >>> phi = HomFGA([[1, 0, 1],
        ...               [0, 1, 1]])
        >>> ker_phi = HomFGA([1, 1, -1])
        >>> (phi * ker_phi) == HomFGA([0, 0])
        True
        >>> phi.compose(ker_phi) == HomFGA([0, 0])
        True
        """

        # If the `other` argument is a numeric type
        if isinstance(other, self._A_entry_types):
            new_A = self.A * other
            return type(self)(new_A, target=self.target, source=self.source)

        if isinstance(other, type(self)):
            if not other.target == self.source:
                raise ValueError('Target of other must equal source of self.')

            new_A = self.A * other.A
            return type(self)(new_A, target=self.target, source=other.source)

        format_args = type(self), type(other)
        raise ValueError('Cannot compose/add {} and {}'.format(*format_args))

    def __mul__(self, other):
        """
        Override the (`*`) operator,
        see :py:meth:`~abelian.morphisms.HomLCA.compose`.
        """
        return self.compose(other)

    def __rmul__(self, other):
        """
        Override the (`*`) operator,
        see :py:meth:`~abelian.morphisms.HomLCA.compose`.
        """
        return self.compose(other)

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
        >>> phi = HomFGA([1])
        >>> psi = HomFGA([2])
        >>> phi.stack_vert(psi) == HomFGA([1, 2])
        True

        """
        if not self.source == other.source:
            raise ValueError('Sources must be equal to stack vertically.')
        new_source = self.source
        new_target = self.target + other.target
        new_A = self.A.col_join(other.A)
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
        >>> phi = HomFGA([1])
        >>> psi = HomFGA([2])
        >>> phi.stack_horiz(psi) == HomFGA([[1, 2]])
        True

        """
        if not self.target == other.target:
            raise ValueError('Targets must be equal to stack horizontally.')
        new_source = self.source + other.source
        new_target = self.target
        new_A = self.A.row_join(other.A)
        return type(self)(new_A, target = new_target, source = new_source)

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
        >>> phi = HomFGA([1])
        >>> psi = HomFGA([2])
        >>> phi.stack_diag(psi) == HomFGA([[1, 0], [0, 2]])
        True

        """
        new_source = self.source + other.source
        new_target = self.target + other.target
        new_A = diag(self.A, other.A)

        return type(self)(new_A, target = new_target, source = new_source)


    def evaluate(self, source_element):
        """
        Apply the morphism to an element.

        Parameters
        ----------
        source_element

        Returns
        -------

        Examples
        ---------
        >>> from sympy import diag
        >>> phi = HomFGA(diag(3, 4), target = [5, 6])
        >>> phi.evaluate([2, 3])
        [1, 0]
        >>> phi.evaluate(Matrix([2, 3]))
        Matrix([
        [1],
        [0]])
        """

        # Project the element to the source LCA of the homomorphism
        source_element = self.source.project_element(source_element)

        # Apply the homomorphism using matrix multiplication, depending
        # on the type of the input
        if isinstance(source_element, Matrix):
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



    def to_HomFGA(self):
        """
        Convert object to HomFGA if possible.

        Returns
        -------
        homomorphism : HomFGA
            The homomorphism converted to a HomFGA instance, if possible.

        Examples
        ---------
        >>> phi = HomLCA([1], source = [1], target = [1])
        >>> isinstance(phi, HomFGA)
        False
        >>> phi = phi.to_HomFGA()
        >>> isinstance(phi, HomFGA)
        True

        """
        integer_entries = all([i % 1 == 0 for i in self.A])
        if self.source.is_FGA() and self.target.is_FGA() and integer_entries:
            return HomFGA(self.A, self.target, self.source)
        else:
            return self


    def equal(self, other):
        """
        Whether or not two HomLCAs are equal.

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
        >>> phi = HomFGA([1], target=[0], source = [0]) # Explicit
        >>> psi = HomLCA([1])   # Shorter, defaults to the above
        >>> phi == psi
        True

        """
        source_equal = self.source == other.source
        target_equal = self.target == other.target
        A_equal = self.A == other.A
        return source_equal and target_equal and A_equal

    def ismomorphic(self, other):
        """
        TODO: Implement this.


        Parameters
        ----------
        other

        Returns
        -------

        """
        pass

    def __eq__(self, other):
        """
        Override the equality (`==`) operator,
        see :py:meth:`~abelian.morphisms.HomLCA.equal`.
        """
        return self.equal(other)

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


    def __repr__(self):
        """
        Override the ``repr()`` function.
        """
        rep = 'source: {}     target: {}\n'.format(repr(self.source),
                                                   repr(self.target))
        rep += str(self.A)
        return rep



class HomFGA(HomLCA):
    """
    A homomorphism between FGAs.
    """

    # The types allowed as entries in A
    _A_entry_types = (int,)

    def project_to_target(self):
        """
        Project columns to target group.

        Returns
        -------
        homomorphism : HomFGA
            A homomorphism with columns projected to the target FGA.

        Examples
        --------
        >>> target = [7, 12]
        >>> phi = HomFGA([[15, 12],
        ...               [9,  17]], target = target)
        >>> phi_proj = HomFGA([[1, 5],
        ...                    [9, 5]], target = target)
        >>> phi.project_to_target() == phi_proj
        True
        """

        A = matrix_mod_vector(self.A, Matrix(self.target.periods))
        return type(self)(A, target = self.target, source = self.source)

    def project_to_source(self):
        """
        Project columns to source group, i.e. compute periods.

        Returns
        -------
        homomorphism : HomFGA
            A homomorphism with periods in the source FGA.

        Examples
        --------
        >>> target = [3, 6]
        >>> phi = HomFGA([[1, 0],
        ...               [3, 3]], target = target)
        >>> phi = phi.project_to_source()
        >>> phi.source.periods == [6, 2]
        True
        """

        # Find dimensions
        m, n = self.A.shape

        # Compute orders for all columns of A
        target_vect = Matrix(self.target.periods)
        source = [order_of_vector(self.A[:, i], target_vect) for i in range(n)]
        return type(self)(self.A, self.target, source)

    def kernel(self):
        """
        Compute the kernel homomorphism.

        Returns
        -------
        homomorphism : HomFGA
            The kernel homomorphism.

        Examples
        --------
        >>> phi = HomFGA([[1, 0, 1], [0, 1, 1]])
        >>> phi.kernel() == HomFGA([-1, -1, 1])
        True

        """
        # Horizontally stack A and ker(pi_2)
        periods = self.target.periods
        A_ker_pi = self.A.row_join(delete_zero_columns(diag(*periods)))

        # Compute SNF, get size and the kernel
        U, S, V = smith_normal_form(A_ker_pi)
        (m, n), r = self.A.shape, len(nonzero_diag_as_list(S))
        kernel_matrix = V[:n, r:]

        kernel = type(self)(kernel_matrix, target = self.source)
        return kernel.project_to_target()

    def cokernel(self):
        """
        Compute the cokernel homomorphism.

        Returns
        -------
        homomorphism : HomFGA
            The cokernel homomorphism.

        Examples
        --------
        >>> phi = HomFGA([[1, 0], [0, 1], [1, 1]])
        >>> coker = phi.cokernel()
        >>> coker.target.isomorphic(LCA([1, 1, 0]))
        True

        """
        # Horizontally stack A and ker(pi_2)
        A_ker_pi = self.A.row_join(delete_zero_columns(diag(
            *self.target.periods)))
        # Compute SNF, get size and the kernel
        U, S, V = smith_normal_form(A_ker_pi)
        diagonal = nonzero_diag_as_list(S)
        (m, n), r = self.A.shape, len(diagonal)
        quotient = diagonal + [0] * (m - r)

        # Initialize the cokernel morphism and project it onto the target
        coker = type(self)(U, target = quotient, source=self.target)
        return coker.project_to_target()

    def image(self):
        """
        Compute the image homomorphism.

        Returns
        -------
        homomorphism : HomFGA
            The image homomorphism.

        Examples
        --------
        >>> phi = HomFGA([[1, 0, 1], [0, 1, 1]])

        """
        # Solve equation for the image
        coim = self.coimage()
        coim_target = Matrix(coim.target.periods)
        solved_mat = solve_epi(coim.A, self.A, coim_target)

        # Initialize morphism and return project onto target
        image = type(self)(solved_mat, self.target, source=coim.target)
        return image.project_to_target()

    def coimage(self):
        """
        Compute the coimage homomorphism.

        Returns
        -------
        homomorphism : HomFGA
            The coimage homomorphism.

        Examples
        --------
        >>> 1 == 1
        True

        """
        # Compute the cokernel of the kernel and return
        kernel = self.kernel()
        coimage = kernel.cokernel()
        return coimage

    def remove_trivial_groups(self):
        """
        Remove trivial from source and target.

        A group is trivial if it is discrete with period 1, i.e. Z_1.
        Removing trivial groups from the target group means removing the
        Z_1 groups from the target, along with the corresponding rows of
        the matrix representing the homomorphism.
        Removing trivial groups from the source group means removing the
        groups Z_1 from the source, i.e. removing every column (generator)
        with period 1.

        Returns
        -------
        homomorphism : HomFGA
            A homomorphism where the trivial groups have been removed from
            the source and the target. The corresponding rows and columns of
            the matrix representing the homomorphism are also removed.

        Examples
        --------
        >>> target = [1, 7]
        >>> phi = HomFGA([[2, 1], [7, 2]], target=target)
        >>> projected = HomFGA([[2]], target=[7], source = [7])
        >>> phi.project_to_source().remove_trivial_groups() == projected
        True

        """
        def trivial(period, discrete):
            return discrete and (period == 1)

        # Get indices where the value of the source is 1
        generator = enumerate(self.source.iterate())
        cols_to_del = [i for (i, (d, p)) in generator if trivial(d, p)]
        new_A = remove_cols(self.A, cols_to_del)

        # Get indices where the value of the target is 1
        generator = enumerate(self.target.iterate())
        rows_to_del = [i for (i, (d, p)) in generator if trivial(d, p)]
        new_A = remove_rows(new_A, rows_to_del)

        new_source = self.source.remove_indices(cols_to_del)
        new_target = self.target.remove_indices(rows_to_del)

        return type(self)(new_A, new_target, new_source)


def Homomorphism(A, target = None, source = None):
    """
    Initiliazes HomFGA or HomLCA depending on inputs.

    This factory function will initialize a HomFGA if both the source and
    target are FGAs, or None. If the source and targets are explicitly given
    as non-discrete LCAs then a HomLCA will be initialized.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix` or list
        A sympy matrix representing the homomorphism. The user may also
        use a list of lists in the form [row1, row2, ...] as input.
    target : LCA or list
        The target of the homomorphism. If None, a discrete target of
        infinite period is used as the default.
    source : LCA or list
        The source of the homomorphism. If None, a discrete source of
        infinite period is used as the default.

    Returns
    -------

    Examples
    --------
    >>> # If no source/target is given, the defalt is discrete (FGA)
    >>> phi = Homomorphism([1])
    >>> isinstance(phi, HomFGA)
    True
    >>> # If the target is continuous, a HomLCA instance will be returned
    >>> target = LCA(periods = [0], discrete = [False])
    >>> phi = Homomorphism([1], target = target)
    >>> isinstance(phi, HomLCA)
    True

    """
    A, target, source = HomLCA._verify_init(A, target, source)
    integer_entries = all([i % 1 == 0 for i in A])
    if target.is_FGA() and source.is_FGA() and integer_entries:
        return HomFGA(A, target=target, source=source)
    else:
        return HomLCA(A, target=target, source=source)





if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = False)


if __name__ == '__main__':
    phi = HomLCA([])
    psi = HomLCA([1])
    print(phi)
    print(psi)
    phi == psi
