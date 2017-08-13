=======
abelian
=======

Welcome to the documentation of ``abelian``, a Python library which facilitates
computations on Locally Compact Abelian groups (LCAs). The LCAs are the groups
:math:`\mathbb{R}`, :math:`\mathbb{T}`, :math:`\mathbb{Z}` and :math:`\mathbb{Z}_n`.
The library is structured into two packages, the ``abelian`` package and the
``abelian.linalg`` sub-package, which is built on the matrix class
:py:class:`~sympy.matrices.dense.MutableDenseMatrix` from the
:py:mod:`sympy` library for symoblic mathematics.

Project Goals
==============

* Classical groups :math:`\mathbb{R}`, :math:`\mathbb{T}`, :math:`\mathbb{Z}` and :math:`\mathbb{Z}_n` and computations on these.
* Relationship between continuous and discrete should be 'pretty'.
* FFT computations on discrete, compact groups (and their products),
  e.g. :math:`\mathbb{Z}_{n_{1}} \oplus \mathbb{Z}_{n_{2}} \oplus \dots \oplus \mathbb{Z}_{n_{r}}`.
* The software should build on the mathematical theory.

.. note::

   The ``abelian`` package is currently being written,
   and is not ready to be used yet.



Software specification
========================

Included below is automatically generated software specification.

.. include:: autodoc_overview.rst


Todo
============

* Create skeleton for project
* Factorizations of homomorphisms between FGAs (tests, docs, implementation)

Contents
========

.. toctree::
   :maxdepth: 2

   License <license>
   Module Reference <api/modules>
   API: abelian <api/abelian>
   API: linalg <api/abelian.linalg>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
