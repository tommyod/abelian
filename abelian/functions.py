#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This module contains ...
"""

from sympy import Matrix
from abelian.linalg.utils import norm
from abelian.utils import call_nested_list, verify_dims_list, argmin, argmax
from types import FunctionType
from abelian.linalg.solvers import solve
import itertools
import numpy as np


class Function:
    """
    A function on a LCA.
    """


    def __init__(self, representation, domain):
        """
        Initialize a function.

        Parameters
        ----------
        representation : function
            The representation is a function.
        domain : LCA
            A locally compact Abelian group for the domain.

        Examples
        ---------

        If a function representation is used, functions on domains are
        straightforward.

        >>> def power(list_arg, exponent = 2):
        ...     return sum(x**exponent for x in list_arg)
        >>> from abelian import Function, LCA
        >>> # A function on R/Z = T
        >>> f = Function(power, LCA([1], [False]))
        >>> f([0.5])
        0.25
        >>> f([1.5], exponent = 3) == 0.5**3
        True
        >>> # A function on Z_p
        >>> f = Function(power, LCA([5, 10]))
        >>> f([1,1]) == f([6, 11])
        True
        >>> f([2, 2], exponent = 1)
        4

        If a table representation is used, the function can be defined on
        direct products of Z_n.

        >>> table = [[1, 2],
        ...          [3, 4],
        ...          [5, 6]]
        >>> f = Function(table, LCA([3, 2]))
        >>> f([1, 1])
        4
        >>> f([3, 1])
        2
        """

        self.domain = domain

        if isinstance(representation, FunctionType):
            # A function representation has been passed
            self.representation = representation
        else:
            # A table representation has been passed
            is_FGA = domain.is_FGA()
            is_periodic = all(p > 0 for p in self.domain.periods)
            if not is_FGA or not is_periodic:
                raise TypeError('When the function representation is a table,'
                                'the domain must be a FGA with finite '
                                'periods.')

            # Verify the dimensions of the data table
            if not verify_dims_list(representation, self.domain.periods):
                raise ValueError('Table dimension mismatch.')

            # Return a callable data table
            def list_caller(list_of_points):
                return call_nested_list(representation, list_of_points)

            self.representation = list_caller
            self.representation.__name__ = 'table'




    def __repr__(self):
        """
        Override the ``repr()`` function.

        Returns
        -------

        """
        f_name = self.representation.__name__
        str = r'Function ({}) on domain {}'.format(f_name, self.domain)
        return str

    def to_latex(self):
        """
        Return as a :math:`\LaTeX` string.


        Returns
        -------

        """
        latex_str = r'\operatorname{FUNC} \in \mathbb{C}^G, \ G = GRP'
        latex_str = latex_str.replace('FUNC', self.representation.__name__)
        latex_str = latex_str.replace('GRP', self.domain.to_latex())
        return latex_str



    def sample(self, list_of_points, *args, **kwargs):
        """
        Sample the function on points.

        Parameters
        ----------
        list_of_points

        Returns
        -------

        """
        # TODO: Make sure sampling is consistent with domain
        # Z should not be sampled at 0.5, 1.5, etc

        return [self.evaluate(p, *args, **kwargs) for p in list_of_points]


    def shift(self, list_shift):
        """
        Shift the function.


        Parameters
        ----------
        list_shift

        Returns
        -------

        """
        new_domain = self.domain

        def new_representation(list_arg, *args, **kwargs):
            """
            A function which first applies the morphism,
            then applies the function.
            """
            generator = zip(list_arg, list_shift)
            shifted_arg = [arg-shift for (arg, shift) in generator]
            applied_func = self.representation(shifted_arg, *args, **kwargs)
            return applied_func

        # Update the name
        new_representation.__name__ = 'shift'
        new_representation = self._update_name(new_representation,
                                               self.representation)

        return type(self)(representation = new_representation,
                          domain = new_domain)

    @staticmethod
    def _update_name(new_func, old_func):
        new_func.__name__ = old_func.__name__ + ' * ' + new_func.__name__
        return new_func




    def evaluate(self, list_arg, *args, **kwargs):
        """
        Evaluate at a point.

        Parameters
        ----------
        list_arg : list
            The first argument, which must be a list (interpreted as vector).
        *args : tuple
            An unpacked tuple of arguments.
        **kwargs : dict
            An unpacked dictionary of arguments.

        Returns
        -------
        value : complex
            A complex number (could be real or integer).

        """

        # If the domain consists of more than one group in the direct sum,
        # the argument must have the same length. If the direct sum consists
        # of one group only and the argument is numeric, we forgive the user
        domain_length = self.domain.length()
        if domain_length > 1:
            if isinstance(list_arg, (int, float, complex)):
                raise ValueError('Argument to function must be list.')
        elif domain_length and isinstance(list_arg, (int, float, complex)):
            list_arg = [list_arg]

        if domain_length != len(list_arg):
            raise ValueError('Function argument does not match domain length.')

        proj_args = self.domain.project_element(list_arg)
        return self.representation(proj_args, *args, **kwargs)

    def __call__(self, list_arg, *args, **kwargs):
        """
        Override function calls,
        see :py:meth:`~abelian.functions.Function.evaluate`.
        """
        return self.evaluate(list_arg, *args, **kwargs)

    def pullback(self, morphism):
        """
        Return the pullback along `morphism`.

        Parameters
        ----------
        morphism : HomLCA
            A homomorphism between LCAs

        Returns
        -------
        pullback : Function
            The pullback of `self` along `morphism`.

        Examples
        --------
        Using a simple function and homomorphism.

        >>> from abelian import Homomorphism, LCA
        >>> # Create a function on Z
        >>> f = Function(lambda list_arg:list_arg[0]**2, LCA([0]))
        >>> # Create a homomorphism from Z to Z
        >>> phi = Homomorphism([2])
        >>> # Pull f back along phi
        >>> f_pullback = f.pullback(phi)
        >>> f_pullback([4]) == 64 # (2*4)**2 == 64
        True

        Using a simple function and homomorphism represented as matrix.

        >>> from abelian import Homomorphism, LCA
        >>> def func(list_arg):
        ...     x, y = tuple(list_arg)
        ...     return x ** 2 + y ** 2
        >>> domain = LCA([5, 3])
        >>> f = Function(func, domain)
        >>> phi = Homomorphism([1, 1], target=domain)
        >>> f_pullback = f.pullback(phi)
        >>>
        >>> f_pullback([8]) == 13
        True
        """

        if not self.domain == morphism.target:
            raise ValueError('Target of morphism must equal domain of '
                             'function.')

        # Get the domain for the new function
        domain = morphism.source


        def new_representation(list_arg, *args, **kwargs):
            """
            A function which first applies the morphism,
            then applies the function.
            """
            applied_morph = morphism.evaluate(list_arg)
            applied_func = self.representation(applied_morph, *args, **kwargs)
            return applied_func

        # Update the name
        new_representation.__name__ = 'pullback'
        new_representation = self._update_name(new_representation,
                                                   self.representation)

        return type(self)(representation = new_representation, domain = domain)




    def pushforward(self, morphism, norm_condition = None):
        """
        Return the pushforward along `morphism`.

        Parameters
        ----------
        morphism

        Returns
        -------

        """
        if not self.domain == morphism.source:
            raise ValueError('Source of morphism must equal domain of '
                             'function.')

        if norm_condition is None:
            def norm_condition(element):
                return norm(element, p = 1) <= 10

        # Get the domain for the new function
        domain = morphism.target


        def new_representation(list_arg, *args, **kwargs):
            """
            A function which first applies the morphism,
            then applies the function.
            """

            # Compute a solution to phi(x) = y
            target_periods = Matrix(morphism.target.periods)
            base_ans = solve(morphism.A, Matrix(list_arg), target_periods)

            # Compute the kernel
            kernel = morphism.kernel()

            # Iterate through the kernel space and compute the sum
            kernel_sum = 0
            dim_ker_source = len(kernel.source)
            vector = list(range(-8, 8))
            for p in itertools.product(*([vector]*dim_ker_source)):

                # The `base_ans` is in the kernel of the morphism,
                # we move to all points in the kernel by taking
                # the `base_ans` + linear combinations of the kernel
                linear_comb = Matrix(list(p))
                kernel_element = base_ans + kernel.evaluate(linear_comb)

                # If the point is not within the norm, continue and
                # do not add it to the sum
                if not norm_condition(kernel_element):
                    continue

                function = self.representation
                func_in_ker = function(kernel_element, *args, **kwargs)
                kernel_sum += func_in_ker


            return kernel_sum

        # Update the name
        new_representation.__name__ = 'pushforward'
        new_representation = self._update_name(new_representation,
                                               self.representation)

        return type(self)(representation=new_representation, domain=domain)


    def transversal(self, epimorphism, transversal_rule,
                    default = 0):
        """
        Pushforward along a transversal.

        Parameters
        ----------
        epimorphism
        transversal_rule

        Returns
        -------

        """
        new_domain = epimorphism.source

        def new_representation(list_arg, *args, **kwargs):
            print('------')
            print(list_arg)
            applied_epi = epimorphism.evaluate(list_arg)
            print(list_arg)
            composed = transversal_rule(applied_epi)
            print(composed)
            if composed == list_arg:
                return self.representation(list_arg, *args, **kwargs)
            else:
                return default

        return type(self)(representation=new_representation, domain=domain)


    def pointwise(self, func, operator):
        """
        Pointwise add/mult/etc.


        Parameters
        ----------
        func
        operator

        Returns
        -------

        """

    def convolve(self, other, norm_cond):
        """
        Convolution.

        Parameters
        ----------
        other

        Returns
        -------

        """

    def idft(self, func_type = ''):
        """
        Compute the inverse discrete fourier transform.

        This is a wrapper around np.fft.ifftn.

        Parameters
        ----------
        func_type : str
            If empty, compute the function values using pure python.
            If 'ogrid', use a numpy.ogrid (open mesh-grid) to compute the
            functino values.
            If 'mgrid', use a numpy.mgrid (dense mesh-grid) to compute the
            function values.

        Returns
        -------
        function : Function
            The inverse discrete Fourier transformation of the original
            function.


        Examples
        --------
        >>> from abelian import LCA, Function
        >>> # Create a simple linear function on Z_5 + Z_4 + Z_3
        >>> domain = LCA([5, 4, 3])
        >>> def linear(list_arg):
        ...     x, y, z = list_arg
        ...     return complex(x + y, z - x)
        >>> func = Function(linear, domain)
        >>> func([1, 2, 1])
        (3+0j)
        >>> func_idft = func.idft()
        >>> func_idft([0, 0, 0])
        (3.5-1j)
        """
        return self._fft_wrapper(func_to_wrap='ifftn', func_type=func_type)

    def dft(self, func_type = ''):
        """
        Compute the discrete fourier transform.

        This is a wrapper around np.fft.fftn.

        Parameters
        ----------
        func_type : str
            If empty, compute the function values using pure python.
            If 'ogrid', use a numpy.ogrid (open mesh-grid) to compute the
            functino values.
            If 'mgrid', use a numpy.mgrid (dense mesh-grid) to compute the
            function values.

        Returns
        -------
        function : Function
            The discrete Fourier transformation of the original function.


        Examples
        --------
        >>> from abelian import LCA, Function
        >>> # Create a simple linear function on Z_5 + Z_4 + Z_3
        >>> domain = LCA([5, 4, 3])
        >>> def linear(list_arg):
        ...     return sum(list_arg)
        >>> func = Function(linear, domain)
        >>> func([1, 2, 1])
        4
        >>> # Take the discrete fourier transform and evaluate
        >>> func_dft = func.dft()
        >>> func_dft([0, 0, 0])
        (270+0j)
        >>> # Take the inverse discrete fourier transform
        >>> func_dft_idft = func_dft.idft()
        >>> # Numerics might not make this equal, but mathematically it is
        >>> abs(func_dft_idft([1, 2, 1]) - func([1, 2, 1])) < 10e-10
        True
        """
        return self._fft_wrapper(func_to_wrap='fftn', func_type=func_type)


    def _fft_wrapper(self, func_to_wrap = 'fftn', func_type = ''):
        """
        Common wrapper for FFT and IFFT routines.

        Parameters
        ----------
        func_to_wrap : str
            Name of the function from the np.fft library to call.
        func_type : str
            If empty, compute the function values using pure python.
            If 'ogrid', use a numpy.ogrid (open mesh-grid) to compute the
            functino values.
            If 'mgrid', use a numpy.mgrid (dense mesh-grid) to compute the
            function values.

        Returns
        -------
        function : Function
            The function with a numpy routine applied to every element
            in the domain.

        """

        # Verify that the inputs are sensible
        domain = self.domain
        dims = domain.periods
        if not all(p > 0 for p in dims) and domain.is_FGA():
            return ValueError('Domain must be discrete and periodic.')

        # Put the function values in a table in preparation for FFT/IFFT
        if func_type == '':
            table = np.empty(dims, dtype=complex)
            for element in itertools.product(*[range(k) for k in dims]):
                table[element] = self.representation(element)
        else:
            # Here the np.ogrid or np.mgrid can be used, see
            # https://arxiv.org/pdf/1102.1523.pdf
            funtion = getattr(np, func_type)
            table = func(funtion[tuple([slice(k) for k in dims])])

        # Take fft and convert to list of lists
        function_wrapped = getattr(np.fft, func_to_wrap, None)
        if function_wrapped is None:
            raise ValueError('Could not wrap:', func_to_wrap)
        table_computed = function_wrapped(table).tolist()

        # Create a new instance and return
        return type(self)(domain = domain, representation = table_computed)










if __name__ == '__main__':
    print('------------------------------')
    from sympy import Matrix, diag
    import math

    from abelian.groups import LCA
    from abelian.morphisms import Homomorphism

    def func(list_arg):
        x, y = tuple(list_arg)
        return 2**(-(x**2 + y**2))

    domain = LCA([0, 0])
    f = Function(func, domain)
    phi = Homomorphism([[1, 0], [0, 2]], target = LCA([2, 3]))

    f_push = f.pushforward(phi)

    ans = f_push([1, 1])

    assert round(ans, 5) == 0.56471

    print('------------------------------')

    def gaussian(list_arg):
        """
        Exponential.
        """
        x = list_arg[0]
        return math.exp(-x**2/2)

    # The domain is Z
    domain = LCA([0])

    # Put the gaussian function on the domain
    f = Function(gaussian, domain)
    print(f)

    plist = list(range(-5, 16))
    # Print some samples
    print('Function defined on Z')
    points = [[k] for k in plist]
    sampled = [round(i, 3) for i in f.sample(points)]
    points = [str(k).ljust(5) for k in points]
    sampled = [str(k).ljust(5) for k in sampled]
    print(*points, sep = '\t')
    print(*sampled, sep='\t')

    # Print some samples
    print('Function defined on Z, shifted')
    f = f.shift([5])
    print(f)
    points = [[k] for k in plist]
    sampled = [round(i, 3) for i in f.sample(points)]
    points = [str(k).ljust(5) for k in points]
    sampled = [str(k).ljust(5) for k in sampled]
    print(*points, sep='\t')
    print(*sampled, sep='\t')


    print('Function moved to Z_10')
    phi = Homomorphism([1], target = [10])
    f = f.pushforward(phi)
    print(f)
    points = [[k] for k in plist]
    sampled = [round(i, 3) for i in f.sample(points)]
    points = [str(k).ljust(5) for k in points]
    sampled = [str(k).ljust(5) for k in sampled]
    print(*points, sep='\t')
    print(*sampled, sep='\t')



    print('------------')
    from abelian import HomLCA
    phi_s = HomLCA([1], LCA([10], [True]), LCA([0], [True]))
    print(phi_s)
    print(phi_s.dual())

    print('------------')
    table = [i**2 for i in range(10)]
    f = Function(table, LCA([10]))
    print(f(11))

    phi = Homomorphism([1], [10])
    f_pulled_to_Z = f.pullback(phi)
    print(f_pulled_to_Z(11))

    epi = Homomorphism([1], [10])
    print(epi)

    def transverse(element):
        element = element[0]
        if abs(element) < abs(10 - element):
            return [element]
        else:
            return [(element - 10)]


    for element in range(10):
        print(element, transverse([element]))

    f_transversed_to_Z = f.transversal(epi, transverse)
    print('transversed')
    x = list(range(-10, 10))
    print(*x, sep = '\t')
    print(*f_transversed_to_Z.sample(x), sep = '\t')



    print('---------------- DFT -----------------------')

    def linear(list_arg):
        return sum(abs(i) for i in list_arg)

    f = Function(linear, LCA([5, 4, 3]))
    print(f)

    f_dft = f.dft()
    print(f_dft([1, 1, 1]), f_dft([0, 0, 0]))













if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = False)





