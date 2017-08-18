Public classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    ~abelian.functions.Function
    ~abelian.morphisms.HomFGA
    ~abelian.morphisms.HomLCA
    ~abelian.groups.LCA

Public functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    ~abelian.morphisms.Homomorphism
    ~abelian.linalg.factorizations.hermite_normal_form
    ~abelian.linalg.factorizations.smith_normal_form
    ~abelian.linalg.solvers.solve

Public classes (detailed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`~abelian.groups.LCA`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

    ~abelian.groups.LCA
    ~abelian.groups.LCA.__add__
    ~abelian.groups.LCA.__eq__
    ~abelian.groups.LCA.__getitem__
    ~abelian.groups.LCA.__init__
    ~abelian.groups.LCA.__iter__
    ~abelian.groups.LCA.__len__
    ~abelian.groups.LCA.__repr__
    ~abelian.groups.LCA.canonical
    ~abelian.groups.LCA.copy
    ~abelian.groups.LCA.dual
    ~abelian.groups.LCA.equal
    ~abelian.groups.LCA.getitem
    ~abelian.groups.LCA.is_FGA
    ~abelian.groups.LCA.isomorphic
    ~abelian.groups.LCA.iterate
    ~abelian.groups.LCA.length
    ~abelian.groups.LCA.project_element
    ~abelian.groups.LCA.rank
    ~abelian.groups.LCA.remove_indices
    ~abelian.groups.LCA.remove_trivial
    ~abelian.groups.LCA.sum
    ~abelian.groups.LCA.to_latex
  
:class:`~abelian.morphisms.HomLCA`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

    ~abelian.morphisms.HomLCA
    ~abelian.morphisms.HomLCA.__add__
    ~abelian.morphisms.HomLCA.__call__
    ~abelian.morphisms.HomLCA.__eq__
    ~abelian.morphisms.HomLCA.__getitem__
    ~abelian.morphisms.HomLCA.__init__
    ~abelian.morphisms.HomLCA.__mul__
    ~abelian.morphisms.HomLCA.__pow__
    ~abelian.morphisms.HomLCA.__radd__
    ~abelian.morphisms.HomLCA.__repr__
    ~abelian.morphisms.HomLCA.__rmul__
    ~abelian.morphisms.HomLCA.add
    ~abelian.morphisms.HomLCA.compose
    ~abelian.morphisms.HomLCA.compose_self
    ~abelian.morphisms.HomLCA.copy
    ~abelian.morphisms.HomLCA.dual
    ~abelian.morphisms.HomLCA.equal
    ~abelian.morphisms.HomLCA.evaluate
    ~abelian.morphisms.HomLCA.getitem
    ~abelian.morphisms.HomLCA.ismomorphic
    ~abelian.morphisms.HomLCA.remove_trivial_groups
    ~abelian.morphisms.HomLCA.stack_diag
    ~abelian.morphisms.HomLCA.stack_horiz
    ~abelian.morphisms.HomLCA.stack_vert
    ~abelian.morphisms.HomLCA.to_HomFGA
    ~abelian.morphisms.HomLCA.to_latex
    ~abelian.morphisms.HomLCA.zero
  
:class:`~abelian.morphisms.HomFGA`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 (inherits from: :class:`~abelian.morphisms.HomLCA` )

.. autosummary::

    ~abelian.morphisms.HomFGA
    ~abelian.morphisms.HomFGA.annihilator
    ~abelian.morphisms.HomFGA.coimage
    ~abelian.morphisms.HomFGA.cokernel
    ~abelian.morphisms.HomFGA.image
    ~abelian.morphisms.HomFGA.kernel
    ~abelian.morphisms.HomFGA.project_to_source
    ~abelian.morphisms.HomFGA.project_to_target
  
:class:`~abelian.functions.Function`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

    ~abelian.functions.Function
    ~abelian.functions.Function.__call__
    ~abelian.functions.Function.__init__
    ~abelian.functions.Function.convolve
    ~abelian.functions.Function.dft
    ~abelian.functions.Function.evaluate
    ~abelian.functions.Function.pointwise
    ~abelian.functions.Function.pullback
    ~abelian.functions.Function.pushfoward
  
