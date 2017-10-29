Public classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    ~abelian.morphisms.HomLCA
    ~abelian.groups.LCA
    ~abelian.functions.LCAFunc

Public functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    ~abelian.linalg.factorizations.hermite_normal_form
    ~abelian.linalg.factorizations.smith_normal_form
    ~abelian.linalg.solvers.solve
    ~abelian.functions.voronoi

Public classes (detailed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`~abelian.functions.LCAFunc`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 (inherits from: :class:`~collections.abc.Callable` )

.. autosummary::

    ~abelian.functions.LCAFunc
    ~abelian.functions.LCAFunc.__call__
    ~abelian.functions.LCAFunc.__init__
    ~abelian.functions.LCAFunc.__repr__
    ~abelian.functions.LCAFunc.copy
    ~abelian.functions.LCAFunc.dft
    ~abelian.functions.LCAFunc.evaluate
    ~abelian.functions.LCAFunc.idft
    ~abelian.functions.LCAFunc.pointwise
    ~abelian.functions.LCAFunc.pullback
    ~abelian.functions.LCAFunc.pushforward
    ~abelian.functions.LCAFunc.sample
    ~abelian.functions.LCAFunc.shift
    ~abelian.functions.LCAFunc.to_latex
    ~abelian.functions.LCAFunc.to_table
    ~abelian.functions.LCAFunc.transversal
  
:class:`~abelian.groups.LCA`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 (inherits from: :class:`~collections.abc.Sequence`, :class:`~collections.abc.Callable` )

.. autosummary::

    ~abelian.groups.LCA
    ~abelian.groups.LCA.__add__
    ~abelian.groups.LCA.__call__
    ~abelian.groups.LCA.__contains__
    ~abelian.groups.LCA.__eq__
    ~abelian.groups.LCA.__getitem__
    ~abelian.groups.LCA.__init__
    ~abelian.groups.LCA.__iter__
    ~abelian.groups.LCA.__len__
    ~abelian.groups.LCA.__pow__
    ~abelian.groups.LCA.__repr__
    ~abelian.groups.LCA.canonical
    ~abelian.groups.LCA.compose_self
    ~abelian.groups.LCA.contained_in
    ~abelian.groups.LCA.copy
    ~abelian.groups.LCA.dual
    ~abelian.groups.LCA.elements_by_maxnorm
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
    ~abelian.groups.LCA.trivial
  
:class:`~abelian.morphisms.HomLCA`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 (inherits from: :class:`~collections.abc.Callable` )

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
    ~abelian.morphisms.HomLCA.annihilator
    ~abelian.morphisms.HomLCA.coimage
    ~abelian.morphisms.HomLCA.cokernel
    ~abelian.morphisms.HomLCA.compose
    ~abelian.morphisms.HomLCA.compose_self
    ~abelian.morphisms.HomLCA.copy
    ~abelian.morphisms.HomLCA.dual
    ~abelian.morphisms.HomLCA.equal
    ~abelian.morphisms.HomLCA.evaluate
    ~abelian.morphisms.HomLCA.getitem
    ~abelian.morphisms.HomLCA.identity
    ~abelian.morphisms.HomLCA.image
    ~abelian.morphisms.HomLCA.kernel
    ~abelian.morphisms.HomLCA.project_to_source
    ~abelian.morphisms.HomLCA.project_to_target
    ~abelian.morphisms.HomLCA.remove_trivial_groups
    ~abelian.morphisms.HomLCA.stack_diag
    ~abelian.morphisms.HomLCA.stack_horiz
    ~abelian.morphisms.HomLCA.stack_vert
    ~abelian.morphisms.HomLCA.to_latex
    ~abelian.morphisms.HomLCA.update
    ~abelian.morphisms.HomLCA.zero
  
