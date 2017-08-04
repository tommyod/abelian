#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Function:
    """
    A function on a LCA.
    """


    def __init__(self, representation, domain):
        """
        Create a function.

        A function is a ....

        Parameters
        ----------
        representation
        domain
        """
        pass


    def call(self, list_arg, *args, **kwargs):
        """
        Evaluate the function.

        Parameters
        ----------
        arg

        Returns
        -------

        """
        pass

    def pullback(self, morphism):
        """
        Pullback.

        Parameters
        ----------
        morphism

        Returns
        -------

        """
        pass

    def pushfoward(self, morphism):
        """
        Pushfoward.

        Parameters
        ----------
        morphism

        Returns
        -------

        """
        pass

    def compose(self, func):
        """
        Compose with C -> C function.

        Parameters
        ----------
        other

        Returns
        -------

        """
        pass

    def pointwise(self, func, operator):
        """
        Pointwise mult/add/... .

        Parameters
        ----------
        func
        operator

        Returns
        -------

        """

    def convolve(self, other):
        """
        Convolution (if domain is discrete + compact).

        Parameters
        ----------
        other

        Returns
        -------

        """


    def dft(self):
        """
        Discrete fourier transform (if domain is discrete + compact).

        Returns
        -------

        """





