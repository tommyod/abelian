#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Function:
    """
    A function on a LCA.
    """


    def __init__(self, representation, domain):
        """
        Create a function.

        Parameters
        ----------
        representation : TODO
        domain : LCA
            A locally compact Abelian group for the domain.

        Examples
        ---------
        """

        self.representation = representation
        self.domain = domain


    def evaluate(self, list_arg, *args, **kwargs):
        """
        Evaluate the function.

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
        if isinstance(list_arg, (int, float, complex)):
            raise ValueError('Argument to function must be list.')

        proj_args = self.domain.project_element(list_arg)
        return self.representation(proj_args)

    def __call__(self, list_arg, *args, **kwargs):
        """
        Override function calls,
        see :py:meth:`~abelian.functions.Function.evaluate`.
        """
        return self.evaluate(list_arg, *args, **kwargs)

    def pullback(self, morphism):
        """
        Return the pullback function, along `morphism`.

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

        return type(self)(representation = new_representation, domain = domain)




    def pushfoward(self, morphism, norm):
        """
        Pushfoward.

        Parameters
        ----------
        morphism

        Returns
        -------

        """
        from abelian.linalg.solvers import solve
        import itertools
        if not self.domain == morphism.source:
            raise ValueError('Source of morphism must equal domain of '
                             'function.')

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
            base_ans = list(base_ans)


            # Compute the kernel
            morphism_kernel = morphism.kernel()

            # Iterate through the kernel space and compute the sum
            answer = 0
            dim_ker_source = len(morphism_kernel.source)
            vector = list(range(-8, 8))
            for p in itertools.product(*([vector]*dim_ker_source)):
                k = morphism_kernel.evaluate(list(p))
                k = Matrix(k)
                print(Matrix(base_ans) + k)
                base_ans_ker = list(Matrix(base_ans) + k)
                a = self.representation(base_ans_ker, *args, **kwargs)
                answer += a
                print(a)


            return answer

        return type(self)(representation=new_representation, domain=domain)





    def pointwise(self, func, operator):
        """
        TODO: Pointwise mult/add/... .

        Parameters
        ----------
        func
        operator

        Returns
        -------

        """

    def convolve(self, other):
        """
        TODO.
        Convolution (if domain is discrete + compact).

        Parameters
        ----------
        other

        Returns
        -------

        """


    def dft(self):
        """
        TODO: The discrete fourier transform.

        Discrete fourier transform (if domain is discrete + compact).

        Returns
        -------

        """





if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = True)

if __name__ == '__main__':
    from sympy import Matrix, diag

    from abelian.groups import LCA
    from abelian.morphisms import Homomorphism

    def func(list_arg):
        x, y = tuple(list_arg)
        return 1/(1 + x**2 + y**2)

    domain = LCA([0, 0])
    f = Function(func, domain)
    phi = Homomorphism([[1, 0], [0, 2]], target = LCA([2, 3]))

    f_push = f.pushfoward(phi)

    ans = f_push([1, 1])

    print(ans)




