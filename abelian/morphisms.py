#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types
from sympy import Matrix, diag, latex
from abelian.utils import mod
from abelian.groups import LCA
from abelian.linalg.utils import delete_zero_columns, nonzero_diag_as_list, \
    matrix_mod_vector, order_of_vector, remove_cols, remove_rows
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
        Write to latex string.

        Returns
        -------

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
        Override the slice (`obj[a:b]`) operator.
        """
        if len(args) != 2:
            raise ValueError('__getitem__ takes 2 arguments.')

        slice1, slice2 = args
        new_A = self.A[slice1, slice2]
        new_target = self.target[slice1]
        new_source = self.source[slice2]
        return type(self)(new_A, new_target, new_source)


    def add(self, other):
        """
        Elementwise addition.

        Parameters
        ----------
        other

        Returns
        -------

        """
        # If the `other` argument is a numeric type
        if isinstance(other, self._A_entry_types):
            m, n = self.A.shape
            new_A = self.A + Matrix(m, n, lambda i,j : other)
            return type(self)(new_A, target=self.target, source=self.source)

        if isinstance(other, type(self)):
            if not self.target == other.target and self.source == other.source:
                raise ValueError('Sources and targets must match..')

            new_A = self.A + other.A
            return type(self)(new_A, target=self.target, source=self.source)

        format_args = type(self), type(other)
        raise ValueError('Cannot add {} and {}'.format(*format_args))

    def __add__(self, other):
        """
        Override the addition (`+`) operator.
        """
        return self.add(other)

    def __radd__(self, other):
        """
        Override the addition (`+`) operator.
        """
        return self.add(other)


    def pow(self, power):
        """
        Repeated composition.

        Parameters
        ----------
        power

        Returns
        -------

        """
        product = self.copy()
        for prod in range(power - 1):
            product *= self
        return product


    def __pow__(self, power, modulo=None):
        """
        Override the power (`**`) operator.
        """
        return self.pow(power)

    def copy(self):
        """
        Return a copy.
        """
        return type(self)(self.A, target = self.target, source = self.source)


    def compose(self, other):
        """
        Compose two homomorphisms.

        Parameters
        ----------
        other

        Returns
        -------
        >>> phi = HomFGA([[1, 0, 1],
        ...               [0, 1, 1]])
        >>> ker_phi = HomFGA([1, 1, -1])
        >>> (phi * ker_phi) == HomFGA([0, 0])
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
        Override the multiplication (`*`) operator.
        """
        return self.compose(other)

    def __rmul__(self, other):
        """
        Override the multiplication (`*`) operator.
        """
        return self.__mul__(other)

    def stack_vert(self, other):
        """
        Stack vertically.

        Parameters
        ----------
        other

        Returns
        -------

        """
        pass

    def stack_horiz(self, other):
        """
        Stack horizontally.

        Parameters
        ----------
        other

        Returns
        -------

        """
        pass

    def stack_diag(self, other):
        """
        Stack diagonally.

        Parameters
        ----------
        other

        Returns
        -------

        """
        pass

    def evaluate(self, source_element):
        """
        Apply the morphism to an element.

        Parameters
        ----------
        source_element

        Returns
        -------

        """
        pass


    def to_HomFGA(self):
        """
        Convert object to HomFGA if possible.

        Returns
        -------

        """
        integer_entries = all([i % 1 == 0 for i in self.A])
        if self.source.is_FGA() and self.target.is_FGA() and integer_entries:
            return HomFGA(self.A, self.target, self.source)
        else:
            return self


    def equal(self, other):
        """
        Check equality.

        Parameters
        ----------
        other

        Returns
        -------

        """
        source_equal = self.source == other.source
        target_equal = self.target == other.target
        A_equal = self.A == other.A
        return source_equal and target_equal and A_equal

    def __eq__(self, other):
        """
        Override the equality (`==`) operator.
        """
        return self.equal(other)


    def __repr__(self):
        """
        Override the `repr()` function.
        """
        rep = 'source: {}     target: {}\n'.format(repr(self.source),
                                                   repr(self.target))
        rep += str(self.A)
        return rep



class HomFGA(HomLCA):
    """
    A homomorphism between two FGAs.
    """

    # The types allowed as entries in A
    _A_entry_types = (int,)

    def project_to_target(self):
        """
        Project columns to target group.

        Returns
        -------

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
        Compute periods.

        Returns
        -------

        Examples
        --------
        >>> target = [3, 6]
        >>> phi = HomFGA([[1, 0],
        ...               [3, 3]], target = target)
        >>> phi = phi.project_to_source()
        >>> phi.source.to_list() == [6, 2]
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
        A_ker_pi = self.A.row_join(delete_zero_columns(diag(*self.target)))

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
        >>> 1 == 1
        True

        """
        # Horizontally stack A and ker(pi_2)
        A_ker_pi = self.A.row_join(delete_zero_columns(diag(*self.target)))
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
        >>> 1 == 1
        True

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
        generator = enumerate(self.source._gen())
        cols_to_del = [i for (i, (d, p)) in generator if trivial(d, p)]
        new_A = remove_cols(self.A, cols_to_del)

        # Get indices where the value of the target is 1
        generator = enumerate(self.target._gen())
        rows_to_del = [i for (i, (d, p)) in generator if trivial(d, p)]
        new_A = remove_rows(new_A, rows_to_del)

        new_source = self.source.delete_by_index(cols_to_del)
        new_target = self.target.delete_by_index(rows_to_del)

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
        return HomFGA(A, source, target)
    else:
        return HomLCA(A, source, target)





if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = True)


if __name__ == '__main__':

    print('Creating from lists')
    target = [7, 12]
    phi = HomFGA([[15, 12], [9, 17]], target=target)
    print(type(phi.source), type(phi.target))
    assert isinstance(phi.source, LCA)
    assert isinstance(phi.target, LCA)

    print('Creating from objects')
    target = LCA([7, 12])
    phi = HomFGA([[15, 12], [9, 17]], target=target)
    print(type(phi.source), type(phi.target))
    assert isinstance(phi.source, LCA)
    assert isinstance(phi.target, LCA)

    print('Creating from matrices')
    target = Matrix([7, 12])
    phi = HomFGA([[15, 12], [9, 17]], target=target)
    print(type(phi.source), type(phi.target))
    assert isinstance(phi.source, LCA)
    assert isinstance(phi.target, LCA)
