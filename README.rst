=======
abelian
=======

.. image:: https://readthedocs.org/projects/abelian/badge/?version=latest
   :target: http://abelian.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

``abelian`` is a Python library for computations on elementary locally compact abelian groups (LCAs).
The elementary LCAs are the groups R, Z, T = R/Z, Z_n and direct sums of these.

.. image:: http://tommyodland.com/abelian/intro_figure.png



Classes and methods
^^^^^^^^^^^^^^^^^^^^^
* The ``LCA`` class represents elementary LCAs, i.e. R, Z, T = R/Z, Z_n and direct sums.
   * Fundamental methods: identity LCA, direct sums, equality, isomorphic, element projection, Pontryagin dual.

* The ``HomLCA`` class represents homomorphisms between LCAs.
   * Fundamental methods: identity morphism, zero morphism, equality, composition, evaluation, stacking, element-wise operations, kernel,    cokernel, image, coimage, dual (adjoint) morphism.

* The ``LCAFunc`` class represents functions from LCAs to complex numbers.
   * Fundamental methods: evaluation, composition, shift (translation), pullback, pushforward, point-wise operators (i.e. addition).


Representation elementary locally compact abelian groups R, Z, T = R/Z and Z_n.


Please see `the documentation <http://abelian.readthedocs.io/en/latest/>`_ for more information. 