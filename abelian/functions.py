#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module consists of a class for functions on LCAs,
called LCAFunc. Such a function represents a function
from a LCA G to the complex numbers C.
"""
from operator import itemgetter
from sympy import Matrix, Float, Integer, Add, Rational
from abelian.linalg import solvers, free_to_free
from abelian.linalg.utils import norm, difference
from abelian.linalg.free_to_free import elements_increasing_norm
from abelian.linalg.solvers import solve
from abelian.utils import call_nested_list, verify_dims_list, copy_func, function_to_table
from abelian.groups import LCA
from types import FunctionType
from collections.abc import Callable
import numpy as np
import functools
import operator


class LCAFunc(Callable):
    """
    A function from an LCA to a complex number.
    """

    def __init__(self, representation, domain):
        """
        Initialize a function G -> C.

        Parameters
        ----------
        representation : function or n-dimensional list of domain allows it
            A function which takes in a list as a first argument, representing
            the group element. Alternatively a list of lists if the domain is
            discrete and of finite order.
        domain : LCA
            An elementary locally compact abelian group, which is the domain
            of the function.

        Examples
        ---------

        If a function representation is used, functions on domains are
        relatively straightforward.

        >>> def power(list_arg, exponent = 2):
        ...     return sum(x**exponent for x in list_arg)
        >>> from abelian import LCAFunc, LCA
        >>> # A function on R/Z = T
        >>> f = LCAFunc(power, LCA([1], [False]))
        >>> f([0.5])
        0.25
        >>> f([1.5], exponent = 3) == 0.5**3
        True
        >>> # A function on Z_p
        >>> f = LCAFunc(power, LCA([5, 10]))
        >>> f([1,1]) == f([6, 11])
        True
        >>> f([2, 2], exponent = 1)
        4

        If a table representation is used, the function can be defined on
        direct sums of Z_n.

        >>> # Define a table: a list of lists
        >>> table = [[1, 2],
        ...          [3, 4],
        ...          [5, 6]]
        >>> f = LCAFunc(table, LCA([3, 2]))
        >>> f([1, 1])
        4
        >>> f([3, 1])
        2
        >>> import numpy as np
        >>> f = LCAFunc(np.array(table), LCA([3, 2]))
        >>> f([1, 1])
        4
        """

        # Verify that the domain is an LCA
        if not isinstance(domain, LCA):
            raise TypeError('Domain must be an LCA instance.')

        self.domain = domain

        # A function representation has been passed
        if isinstance(representation, FunctionType):
            # A function representation has been passed
            self.representation = representation
            self.table = None

        # A table representation has been passed
        else:
            if not self._discrete_finite_domain():
                raise TypeError('When the function representation is a table,'
                                'the domain must be a FGA with finite '
                                'orders.')

            # Verify the dimensions of the data table
            if not verify_dims_list(representation, self.domain.orders):
                raise ValueError('Table dimension mismatch.')

            # Return a callable data table
            def list_caller(list_of_points):
                return call_nested_list(representation, list_of_points)

            self.representation = list_caller
            self.table = representation


    def __call__(self, list_arg, *args, **kwargs):
        """
        Override function calls,
        see :py:meth:`~abelian.functions.LCAFunc.evaluate`.
        """
        return self.evaluate(list_arg, *args, **kwargs)

    def __repr__(self):
        """
        Override the ``repr()`` function.

        Returns
        -------
        representation :str
            A representation of the instance.

        """
        str = r'LCAFunc on domain {}'.format(self.domain)
        return str

    def copy(self):
        """
        Return a copy of the instance.

        Returns
        -------
        function : LCAFunc
            A copy of `self`.

        Examples
        --------
        >>> from abelian import LCA, LCAFunc
        >>> f = LCAFunc(lambda x:sum(x), LCA([0]))
        >>> g = f.copy()
        >>> f([1]) == g([1])
        True
        """
        repr = copy_func(self.representation)
        domain = self.domain.copy()
        return type(self)(representation = repr, domain = domain)

    def dft(self, func_type = None):
        """
        If the domain allows it, compute DFT.

        This method uses the n-dimensional Fast Fourier Transform (FFT) to
        compute the n-dimensional Discrete Fourier Transform. The data is
        converted to a :py:class:`~numpy.ndarray` object for efficient
        numerical computation, then the :py:func:`~numpy.fft.fftn` function
        is used to compute the fast fourier transform.

        This implementation is different from the implementation in
        :py:func:`~numpy.fft.fftn` by a factor. While the :py:func:`~numpy.fft.fftn`
        function divides by m*n on the inverse transform, this implementation
        does it on the forward transform, and vice verca.


        Parameters
        ----------
        func_type : str
            If None, compute the function values using pure python.
            If 'ogrid', use a numpy.ogrid (open mesh-grid) to compute the
            functino values.
            If 'mgrid', use a numpy.mgrid (dense mesh-grid) to compute the
            function values.

        Returns
        -------
        function : LCAFunc
            The discrete Fourier transformation of the original function.


        Examples
        --------
        >>> from abelian import LCA, LCAFunc
        >>> # Create a simple linear function on Z_5 + Z_4 + Z_3
        >>> domain = LCA([5, 4, 3])
        >>> def linear(list_arg):
        ...     return sum(list_arg)
        >>> func = LCAFunc(linear, domain)
        >>> func([1, 2, 1])
        4
        >>> # Take the discrete fourier transform and evaluate
        >>> func_dft = func.dft()
        >>> func_dft([0, 0, 0])
        (4.5+0j)
        >>> # Take the inverse discrete fourier transform
        >>> func_dft_idft = func_dft.idft()
        >>> # Numerics might not make this equal, but mathematically it is
        >>> abs(func_dft_idft([1, 2, 1]) - func([1, 2, 1])) < 10e-10
        True
        """
        return self._fft_wrapper(func_to_wrap='fftn', func_type=func_type)



    def evaluate(self, list_arg, *args, **kwargs):
        """
        Evaluate function on a group element.

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
        value : complex, float or int
            A complex number (could be real or integer).

        Examples
        --------
        >>> from abelian import LCA, LCAFunc
        >>> R = LCA([0], [False])
        >>> function = LCAFunc(lambda x: 1, domain = R**2)
        >>> function([1, 2])
        1

        Some subtle concepts are shown below.

        >>> function(1)
        Traceback (most recent call last):
        ...
        ValueError: Argument to function must be list.
        >>> function([1])
        Traceback (most recent call last):
        ...
        ValueError: LCAFunc argument does not match domain length.
        >>> type(function([1, 1])) in (int, float, complex)
        True
        """

        # If the domain consists of more than one group in the direct sum,
        # the argument must have the same length. If the direct sum consists
        # of one group only and the argument is numeric, we forgive the user
        domain_length = self.domain.length()

        # Verify the inputs
        if domain_length > 1:
            if isinstance(list_arg, (int, float, complex)):
                raise ValueError('Argument to function must be list.')
        elif domain_length and isinstance(list_arg, (int, float, complex)):
            list_arg = [list_arg]

        # Verify the length
        if domain_length != len(list_arg):
            raise ValueError('LCAFunc argument does not match domain length.')

        # Project and compute
        proj_args = self.domain.project_element(list_arg)
        answer = self.representation(proj_args, *args, **kwargs)

        # Cast to Python data type
        if isinstance(answer, (Float, Rational)):
            return float(answer)
        if isinstance(answer, Integer):
            return int(answer)
        if isinstance(answer, Add):
            return complex(answer)
        return answer

    def idft(self, func_type = None):
        """
        If the domain allows it, compute inv DFT.

        This is a wrapper around np.fft.ifftn.

        Parameters
        ----------
        func_type : str
            If None, compute the function values using pure python.
            If 'ogrid', use a numpy.ogrid (open mesh-grid) to compute the
            functino values.
            If 'mgrid', use a numpy.mgrid (dense mesh-grid) to compute the
            function values.

        Returns
        -------
        function : LCAFunc
            The inverse discrete Fourier transformation of the original
            function.


        Examples
        --------
        >>> from abelian import LCA, LCAFunc
        >>> # Create a simple linear function on Z_5 + Z_4 + Z_3
        >>> domain = LCA([5, 4, 3])
        >>> def linear(list_arg):
        ...     x, y, z = list_arg
        ...     return complex(x + y, z - x)
        >>> func = LCAFunc(linear, domain)
        >>> func([1, 2, 1])
        (3+0j)
        >>> func_idft = func.idft()
        >>> func_idft([0, 0, 0])
        (210-60j)
        """
        return self._fft_wrapper(func_to_wrap='ifftn', func_type=func_type)


    def pointwise(self, other, operator):
        """
        Apply pointwise binary operator.

        Parameters
        ----------
        other : LCAFunc
            Another Functin on the same domain.
        operator : function
            A binary operator.

        Returns
        -------
        function : LCAFunc
            The resulting function, new = operator(self, other).

        Examples
        --------
        >>> from abelian import LCA
        >>> domain = LCA([5])
        >>> function1 = LCAFunc(lambda arg: sum(arg), domain)
        >>> function2 = LCAFunc(lambda arg: sum(arg)*2, domain)
        >>> from operator import add
        >>> pointwise_add = function1.pointwise(function2, add)
        >>> function1([2]) + function2([2]) == pointwise_add([2])
        True
        >>> from operator import mul
        >>> sample_points = [0, 1, 2, 3, 4]
        >>> pointwise_mul = function1.pointwise(function2, mul)
        >>> pointwise_mul.sample(sample_points) # i * 2*i = 2*i*i
        [0, 2, 8, 18, 32]

        """
        if self.domain != other.domain:
            raise ValueError('Domains must be equal.')

        # Perform both function and apply the operator
        def new_repr(list_arg, *args, **kwargs):
            result_self = self.representation(list_arg, *args, **kwargs)
            result_other = other.representation(list_arg, *args, **kwargs)
            return operator(result_self, result_other)

        return type(self)(domain = self.domain, representation = new_repr)

    def pullback(self, morphism):
        """
        Return the pullback along `morphism`.

        The pullback is the composition `morphism`, then `self`.
        The domain of `self` must match the target of the morphism.

        Parameters
        ----------
        morphism : HomLCA
            A homomorphism between LCAs

        Returns
        -------
        pullback : LCAFunc
            The pullback of `self` along `morphism`.

        Examples
        --------
        Using a simple function and homomorphism.

        >>> from abelian import HomLCA, LCA
        >>> # Create a function on Z
        >>> f = LCAFunc(lambda list_arg:list_arg[0]**2, LCA([0]))
        >>> # Create a homomorphism from Z to Z
        >>> phi = HomLCA([2])
        >>> # Pull f back along phi
        >>> f_pullback = f.pullback(phi)
        >>> f_pullback([4]) == 64 # (2*4)**2 == 64
        True

        Using a simple function and homomorphism represented as matrix.

        >>> from abelian import HomLCA, LCA
        >>> def func(list_arg):
        ...     x, y = tuple(list_arg)
        ...     return x ** 2 + y ** 2
        >>> domain = LCA([5, 3])
        >>> f = LCAFunc(func, domain)
        >>> phi = HomLCA([1, 1], target=domain)
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

        def new_repr(list_arg, *args, **kwargs):
            """
            A function which first applies the morphism,
            then applies the function.
            """
            applied_morph = morphism.evaluate(list_arg)
            applied_func = self.representation(applied_morph, *args, **kwargs)
            return applied_func

        return type(self)(representation = new_repr, domain = domain)

    def pushforward(self, morphism, terms_in_sum = 50):
        """
        Return the pushforward along `morphism`.

        The pushforward is computed by solving an equation, finding the
        kernel, and iterating through the kernel. The pushfoward
        approximates a possibly infinite sum by `terms_in_sum` terms.


        Parameters
        ----------
        morphism : HomLCA
            A homomorphism between LCAs.
        terms_in_sum : int
            The number of terms in the sum to use, i.e. the number of solutions
            to the equation to iterate over.
        norm_condition : function
            If not None, a function can be used to terminate the sum.
            The norm_condition must be a function of a group element,
            and when the function is false for every v in the kernel
            such that maxnorm(v) = C for a given C, then the sum terminates.


        Returns
        -------
        pushforward : LCAFunc
            The pushforward of `self` along `morphism`.

        Examples
        --------

        The first example is a homomorphism R -> T.

        >>> from abelian import LCA, LCAFunc, HomLCA
        >>> R = LCA([0], [False])
        >>> T = LCA([1], [False])
        >>> epimorphism = HomLCA([1], source = R, target = T)
        >>> func_expr = lambda x: 2**-sum(x_j**2 for x_j in x)
        >>> func = LCAFunc(func_expr, domain = R)
        >>> func.pushforward(epimorphism, 1)([0]) # 1 term in the sum
        1.0
        >>> func.pushforward(epimorphism, 3)([0]) # 1 + 0.5*2
        2.0
        >>> func.pushforward(epimorphism, 5)([0]) # 1 + 0.5*2 + 0.0625*2
        2.125

        The first example is a homomorphism Z -> Z_2.

        >>> from abelian import LCA, LCAFunc, HomLCA
        >>> Z = LCA([0], [True])
        >>> Z_2 = LCA([2], [True])
        >>> epimorphism = HomLCA([1], source = Z, target = Z_2)
        >>> func_expr = lambda x: 2**-sum(x_j**2 for x_j in x)
        >>> func = LCAFunc(func_expr, domain = Z)
        >>> func.pushforward(epimorphism, 1)([0]) # 1 term in the sum
        1.0
        >>> func.pushforward(epimorphism, 3)([0]) # 1 + 0.5*2 + 0.0625*2
        1.125

        The third example is a homomorphism R -> R.

        >>> from abelian import LCA, LCAFunc, HomLCA
        >>> R = LCA([0], [False])
        >>> epimorphism = HomLCA([1], source = R, target = R)
        >>> func_expr = lambda x: 2**-sum(x_j**2 for x_j in x)
        >>> func = LCAFunc(func_expr, domain = R)
        >>> func.pushforward(epimorphism, 3)([0]) # 1 term in the sum
        1.0


        """
        if not self.domain == morphism.source:
            raise ValueError('Source of morphism must equal domain of '
                             'function.')

        # Compute the kernel
        kernel = morphism.kernel()
        kernel_m, kernel_n = kernel.A.shape

        # Get the domain for the new function
        domain = morphism.target

        def new_representation(list_arg, *args, **kwargs):
            """
            A function which first applies the morphism,
            then applies the function.
            """
            # Compute a solution to phi(x) = y
            target_orders = Matrix(morphism.target.orders)
            base_ans = solve(morphism.A, Matrix(list_arg), target_orders)

            # The kernel is empty, do not start summing, just return
            if kernel_n == 0:
                return self.representation(base_ans, *args, **kwargs)

            # Iterate through the kernel space and compute the sum
            kernel_sum = 0
            dim_ker_source = len(kernel.source)

            generator = elements_increasing_norm(dim_ker_source)
            for counter, boundary_element in enumerate(generator, 1):
                # The `base_ans` is in the kernel of the morphism,
                # we move to all points in the kernel by taking
                # the `base_ans` + linear combinations of the kernel
                linear_comb = Matrix(list(boundary_element))
                kernel_element = base_ans + kernel.evaluate(linear_comb)

                function = self.representation
                func_in_ker = function(kernel_element, *args, **kwargs)
                kernel_sum += func_in_ker
                if counter >= terms_in_sum:
                    break

            return kernel_sum

        return type(self)(representation=new_representation, domain=domain)


    def sample(self, list_of_elements, *args, **kwargs):
        """
        Sample on a list of group elements.

        Parameters
        ----------
        list_of_elements : list
            A list of groups elements, where each element is also a list.

        Returns
        -------
        sampled_vals : list
            A list of sampled values at the elements.

        Examples
        --------
        >>> from abelian import LCAFunc, LCA
        >>> func = LCAFunc(lambda x : sum(x), LCA([0, 0]))
        >>> sample_points = [[0, 0], [1, 2], [2, 1], [3, 3]]
        >>> func.sample(sample_points)
        [0, 3, 3, 6]
        """

        return [self.evaluate(p, *args, **kwargs) for p in list_of_elements]


    def shift(self, list_shift):
        """
        Shift the function.


        Parameters
        ----------
        list_shift : list
            A list of shifts.

        Returns
        -------
        function : LCAFunc
            A new function which is shifted.

        Examples
        --------
        >>> from abelian import LCAFunc, LCA
        >>> func = LCAFunc(lambda x: sum(x), LCA([0]))
        >>> func.sample([0, 1, 2, 3])
        [0, 1, 2, 3]
        >>> func.shift([2]).sample([0, 1, 2, 3])
        [-2, -1, 0, 1]
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


        return type(self)(representation = new_representation,
                          domain = new_domain)


    def to_latex(self):
        """
        Return as a :math:`\LaTeX` string.


        Returns
        -------
        latex_str : str
            The object as a latex string.

        """
        latex_str = r'\operatorname{function} \in \mathbb{C}^G, \ G = GRP'
        latex_str = latex_str.replace('GRP', self.domain.to_latex())
        return latex_str

    def to_table(self, *args, **kwargs):
        """
        Return a n-dimensional table.

        Returns
        -------
        table : n-dimensional list
            The table representation.

        Examples
        --------
        >>> from abelian import LCA, LCAFunc
        >>> domain = LCA([5, 5])
        >>> f = LCAFunc(lambda x: sum(x), domain)
        >>> table = f.to_table()
        >>> table[1][1]
        (2+0j)

        Using a table from the start.

        >>> from abelian import LCA, LCAFunc
        >>> import numpy as np
        >>> domain = LCA([5, 5])
        >>> f = LCAFunc(np.eye(5), domain)
        >>> table = f.to_table()
        >>> table[1][1]
        1.0
        >>> type(table)
        <class 'numpy.ndarray'>
        >>> f = LCAFunc([[1, 2], [2, 4]], LCA([2, 2]))
        >>> f.to_table()
        [[1, 2], [2, 4]]
        """

        # If the domain is not discrete and of finite order, no table exists
        if not self._discrete_finite_domain():
            raise TypeError('No table. Domain must be discrete and finite.')

        # If a table already is computed, return it
        if self.table is not None:
            return self.table

        # If a table is not computed, compute it and return
        dims = self.domain.orders
        table = function_to_table(self.representation, dims, *args, **kwargs)
        self.table = table
        return table

    def transversal(self, epimorphism, transversal_rule = None,
                    default_value = 0):
        """
        Pushforward using transversal rule.

        If (transversal * epimorphism)(x) = x, then x is pushed forward
        using the transversal rule. If not, then the default_value value is
        returned.

        Parameters
        ----------
        epimorphism : HomLCA
            An epimorphism.
        transversal_rule : function
            A function with signature `func(list_arg, *args, **kwargs)`.

        Returns
        -------
        function : LCAFunc
            The pushforward of `self` along the transversal of the epimorphism.

        Examples
        --------
        >>> from abelian import LCA, LCAFunc, HomLCA
        >>> n = 5 # Sice of the domain, Z_n
        >>> f_on_Zn = LCAFunc(lambda x: sum(x)**2, LCA([n]))
        >>> # To move this function to Z, create an epimorphism and a
        >>> # transversal rule
        >>> epimorphism = HomLCA([1], source = [0], target = [n])
        >>> def transversal_rule(x):
        ...     if sum(x) < n/2:
        ...         return [sum(x)]
        ...     elif sum(x) >= n/2:
        ...         return [sum(x) - n]
        ...     else:
        ...         return None
        >>> # Do the pushforward with the transversal rule
        >>> f_on_Z = f_on_Zn.transversal(epimorphism, transversal_rule)
        >>> f_on_Z.sample(list(range(-n, n+1)))
        [0, 0, 0, 9.0, 16.0, 0.0, 1.0, 4.0, 0, 0, 0]

        """
        new_domain = epimorphism.source

        if not transversal_rule:
            transversal_rule = voronoi(epimorphism, norm_p = 2)

        def new_representation(list_arg, *args, **kwargs):
            # Compose (section * transversal)(x)
            applied_epi = epimorphism.evaluate(list_arg)
            composed = transversal_rule(applied_epi)

            # If the composition is the identity, apply the epimorphism
            # and then the function to evaluate the new function at the point
            epsilon = 10e-10
            if difference(composed,list_arg) < epsilon:#composed == list_arg:
                return self.representation(applied_epi, *args, **kwargs)
            else:
                return default_value

        return type(self)(representation=new_representation, domain=new_domain)

    def _discrete_finite_domain(self):
        """
        Whether or not the domain is discrete and of finite order.

        Returns
        -------
        discrete_finite : bool
            Whether or not the domain is discrete and finite.
        """
        return self.domain.is_FGA() and all(p > 0 for p in self.domain.orders)


    def _fft_wrapper(self, func_to_wrap = 'fftn', func_type = ''):
        """
        Common wrapper for FFT and IFFT routines.

        The numpy DFT is defined as:
        :math:`A_{kl} =  \sum_{m=0}^{M-1} \sum_{n=0}^{N-1}
        a_{mn}\exp\left\{-2\pi i \left({mk\over M}+{nl\over N}\right)\right\}
        \qquad k = 0, \ldots, M-1;\quad l = 0, \ldots, N-1.`

        And the inverse DFT is defined as:
        :math:`a_{mn} = \frac{1}{MN} \sum_{k=0}^{M-1} \sum_{l=0}^{N-1}
        A_{kl}\exp\left\{2\pi i \left({mk\over M}+{nl\over N}\right)\right\}
        \qquad m = 0, \ldots, M-1;\quad n = 0, \ldots, N-1.`


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
        function : LCAFunc
            The function with a numpy routine applied to every element
            in the domain.

        """

        # Verify that the inputs are sensible
        domain = self.domain
        dims = domain.orders
        if not all(p > 0 for p in dims) and domain.is_FGA():
            return ValueError('Domain must be discrete and of finite order.')

        # Put the function values in a table in preparation for FFT/IFFT
        if func_type is None:
            table = self.to_table()
        else:
            # Here the np.ogrid or np.mgrid can be used, see
            # https://arxiv.org/pdf/1102.1523.pdf
            function = getattr(np, func_type)
            table = function([tuple([slice(k) for k in dims])])

        # Take fft and convert to list of lists
        function_wrapped = getattr(np.fft, func_to_wrap, None)
        if function_wrapped is None:
            raise ValueError('Could not wrap:', func_to_wrap)
        table_computed = function_wrapped(table)

        # Scale differently then the Numpy implementation
        # Numpy divides by prod(dims) when computing the inverse,
        # but we do it when we compute the forward transform
        if func_to_wrap == 'fftn':
            table_computed =  table_computed / (functools.reduce(
                operator.mul, dims))
        elif func_to_wrap == 'ifftn':
            table_computed =  table_computed * (functools.reduce(
                operator.mul, dims))

        # Create a new instance and return
        return type(self)(domain = domain, representation = table_computed)


def voronoi(epimorphism, norm_p=2):
    """
    Return the Voronoi transversal function.

    This higher-order function returns a quotient transversal
    which maps x to the y which is cloest to the low-frequency
    fourier mode.

    Parameters
    ----------
    epimorphism_kernel : HomLCA
        The kernel of the epimorphism that we want to find a section for.
    norm : function or None
        A norm function, if None, the max-norm is used.

    Returns
    -------
    sigma : function
        A function x -> y.

    Examples
    ---------
    >>> # An orthogonal example
    >>> from abelian import LCAFunc, HomLCA, LCA
    >>> Z_10 = LCA([10])
    >>> epimorphism = HomLCA([1], target = Z_10)
    """

    # Verify that the epimorphism is R^n -> T^n or Z^n -> Z_n
    if not all(o == 0 for o in epimorphism.source.orders):
        raise ValueError('Epimorphism must be R^n -> T^n or Z^n -> Z_n')
    if not all(o >= 1 for o in epimorphism.target.orders):
        raise ValueError('Epimorphism must be R^n -> T^n or Z^n -> Z_n')

    # Solve for the kernel of the epimorphism
    epimorphism_kernel = epimorphism.kernel()

    def sigma(x):
        """
        A section used for quotient transversal.
        """
        # Initialize variables
        x = Matrix(x)
        kernel_A = epimorphism_kernel.A
        m, m = kernel_A.shape

        # STEP 1: Solve the equation phi(y) = x for the y
        # If the epimorphism is Z^n -> Z_n
        if epimorphism._is_homFGA():
            # Use the solver for FGAs
            p = Matrix(epimorphism.target.orders)
            y = solvers.solve(epimorphism.A, x, p = p)

        # If R^n -> T^n
        if (all(d == False for d in epimorphism.source.discrete) and
            all(d == False for d in epimorphism.target.discrete)):
            # Normal linear algebra, user sympy solver
            y = epimorphism.A.solve(x)

        # STEP 2: Find the points with max-norm <= 1
        points = free_to_free.elements_increasing_norm(m, end_value = 2)
        points = (Matrix(p) for p in points)

        # STEP 3: Find the corner that minimizes ||y - ker(epi) * points||
        funcvals_vals = ((y - kernel_A*p, p) for p in points)

        normvals_vals = ((norm(list(val), norm_p), p) for (val, p) in funcvals_vals)
        min_value, minimizer = min(normvals_vals, key=itemgetter(0))

        y = y - kernel_A*minimizer

        return list(y)

    return sigma


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = False)