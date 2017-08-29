=======
abelian
=======

Project overview
------------------

Short description
~~~~~~~~~~~~~~~~~~~~

Welcome to the documentation of ``abelian``, a Python library which facilitates
computations on Locally Compact Abelian groups (LCAs). The LCAs are groups
isomorphic to :math:`\mathbb{R}`, :math:`T = \mathbb{R}/\mathbb{Z}`, :math:`\mathbb{Z}`,
:math:`\mathbb{Z}_n` and direct sums of these.
The library is structured into two packages, the ``abelian`` package and the
``abelian.linalg`` sub-package, which is built on the matrix class
:py:class:`~sympy.matrices.dense.MutableDenseMatrix` from the
:py:mod:`sympy` library for symbolic mathematics.


Project goals
~~~~~~~~~~~~~~~~~~~~

* Represent the groups :math:`\mathbb{R}`, :math:`T`,
  :math:`\mathbb{Z}` and :math:`\mathbb{Z}_n` and facilitate computations on
  these.
* Relationship between continuous and discrete should be 'pretty'.
* FFT computations on discrete, finite groups and their products.
* The software should build on the mathematical theory.


.. note::

   The ``abelian`` package is currently being written,
   and is not ready to be used yet.

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
