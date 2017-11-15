API
==========

Library structure
--------------------------------


The ``abelian`` library consists of two packages, ``abelian`` and the
``abelian.linalg`` sub-package.

* ``abelian`` - Provides access to high-level mathematical objects: LCAs,
  homomorphisms between LCAs and functions from an LCA to the complex numbers.

   * ``abelian.linalg`` - Lower-level linear algebra routines. Most notably
     the Hermite normal form, the Smith normal form, an equation solver for
     the equation `Ax = b mod p` over the integers, as well as functions for
     generating elements of a finitely generated abelian group (FGA) ordered
     by maximum-norm.


Full API
--------------------------------

.. toctree::
   :maxdepth: 2

    Module Reference <api/modules>
    API: abelian <api/abelian>
    API: linalg <api/abelian.linalg>