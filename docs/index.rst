=======
abelian
=======

Project overview
------------------

Short description
~~~~~~~~~~~~~~~~~~~~

Welcome to the documentation of ``abelian``, a Python library which facilitates
computations on elementary locally compact abelian groups (LCAs). The LCAs are
the groups isomorphic to
:math:`\mathbb{R}`,
:math:`T = \mathbb{R}/\mathbb{Z}`,
:math:`\mathbb{Z}`,
:math:`\mathbb{Z}_n`
and direct sums of these.
The library is structured into two packages, the ``abelian`` package and the
``abelian.linalg`` sub-package, which is built on the matrix class
:py:class:`~sympy.matrices.dense.MutableDenseMatrix` from the
:py:mod:`sympy` library for symbolic mathematics.


Classes and methods
^^^^^^^^^^^^^^^^^^^^^
* The :py:class:`~abelian.LCA` class
  represents elementary LCAs.

   * **Fundamental methods**: identity LCA, direct sums, equality, isomorphic,
     element projection, Pontryagin dual.

* The :py:class:`~abelian.HomLCA` class represents homomorphisms between
  LCAs.

   * **Fundamental methods**: identity morphism, zero morphism, equality,
     composition, evaluation, stacking, element-wise operations, kernel,
     cokernel, image, coimage, dual (adjoint) morphism.

* The :py:class:`~abelian.LCAFunc` class represents functions from LCAs to
  complex numbers.

   * **Fundamental methods**: evaluation, composition, shift (translation),
     pullback, pushforward, point-wise operators (i.e. addition).

Algorithms for the Smith normal form and Hermite normal form are also
implemented in
:py:func:`~abelian.linalg.factorizations.smith_normal_form`
and
:py:func:`~abelian.linalg.factorizations.hermite_normal_form`
respectively.

Project goals
~~~~~~~~~~~~~~~~~~~~

* Represent the groups :math:`\mathbb{R}`, :math:`T`,
  :math:`\mathbb{Z}` and :math:`\mathbb{Z}_n` and facilitate computations on
  these.
* Relationship between continuous and discrete should be 'pretty'.
* DFT computations on discrete, finite groups and their products using the FFT.
* The software should build on the mathematical theory.


Installation
~~~~~~~~~~~~~~~~~~~~

(1) Download the Anaconda_ distribution of Python_, version 3.X.
(2) Depending on your operating system, do one of the following:

    (a) If on **Windows**, open the Anaconda prompt and run
        ``pip install abelian`` to install ``abelian``
        from PyPI_.
    (b) If on **Linux** og **Mac**, open the terminal and run
        ``pip install abelian`` to install ``abelian``
        from PyPI_.

(3) Open a Python editor (such as Spyder_, which comes with Anaconda_)
    and type ``from abelian import *`` to import all classes and functions
    from the library. You're all set, go try some examples
    from the :doc:`tutorials <notebooks/tutorials>`.


.. _Anaconda: https://www.continuum.io/downloads
.. _Python: https://www.python.org/
.. _PyPI: https://pypi.org/project/abelian/
.. _Spyder: https://pythonhosted.org/spyder/


Contents
------------------

.. toctree::
   :maxdepth: 2

   Software Specification <software_specification>
   Tutorials <notebooks/tutorials>
   API <api>

Indices and tables
--------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
