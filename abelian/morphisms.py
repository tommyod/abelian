#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sympy import Matrix, diag
from abelian.utils import mod
from abelian.groups import LCA
from abelian.linalg.utils import delete_zero_columns, nonzero_diag_as_list, \
    matrix_mod_vector, order_of_vector
from abelian.linalg.factorizations import smith_normal_form


class HomLCA:
    """
    Class for homomorphisms between LCAs.
    """

    def __init__(self, A, target = None, source = None):
        """
        Initialize a homomorphism between two LCAs.

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
        self.A = A
        self.target = target
        self.source = source

    @staticmethod
    def _verify_init(A, target, source):
        """
        Verify the inputs.
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

        # If lists were passed as source/targets, convert to LCAs
        if isinstance(target, list):
            target = LCA(target)
        if isinstance(source, list):
            source = LCA(source)

        return A, target, source

    def compose(self, other):
        """
        Compose two homomorphisms.

        Parameters
        ----------
        other

        Returns
        -------

        """
        pass

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

        Parameters
        ----------
        other

        Returns
        -------

        """
        return self.equal(other)


    def __repr__(self):
        """
        Representation.
        """
        rep = 'source: {}     target: {}\n'.format(repr(self.source),
                                                   repr(self.target))
        rep += str(self.A)
        return rep



class HomFGA(HomLCA):
    """
    Class for homomorphisms between FGAs.
    """

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
        source = [order_of_vector(self.A[:, i], self.target) for i in range(n)]
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
        >>> 1 == 1
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
        pass

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
        pass

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
        pass


def Homomorphism(A, target = None, source = None):
    """
    Initiliazes HomFGA or HomLCA depending on inputs.

    This factor function will initialize a HomFGA if both the source and
    target are FGAs, or None. If the source and targets are explicitly given
    as non-discrete LCAs then a HomLCA will be initialized.

    Parameters
    ----------
    A
    target
    source

    Returns
    -------

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
    target = [7, 12]
    phi = HomFGA([[15, 12], [9, 17]], target=target)
    phi_proj = HomFGA([[1, 5], [9, 5]], target=target)
    print(phi.project_to_target())
    print(phi_proj)
    print(phi.project_to_target() == phi_proj)