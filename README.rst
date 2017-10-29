=======
abelian
=======

.. image:: https://readthedocs.org/projects/abelian/badge/?version=latest
   :target: http://abelian.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

``abelian`` is a Python library for computations on elementary locally compact abelian groups (LCAs).
The elementary LCAs are the groups R, Z, T = R/Z, Z_n and direct sums of these.
The Fourier transformation is defined on these groups.
Using ``abelian``, it is possible to sample, periodize and do Fourier analysis on elementary LCAs using group theory.

.. image:: http://tommyodland.com/abelian/intro_figure.png



Classes and methods
^^^^^^^^^^^^^^^^^^^^^
* The ``LCA`` class represents elementary LCAs, i.e. R, Z, T = R/Z, Z_n and direct sums.
   * Fundamental methods: identity LCA, direct sums, equality, isomorphic, element projection, Pontryagin dual.

* The ``HomLCA`` class represents homomorphisms between LCAs.
   * Fundamental methods: identity morphism, zero morphism, equality, composition, evaluation, stacking, element-wise operations, kernel,    cokernel, image, coimage, dual (adjoint) morphism.

* The ``LCAFunc`` class represents functions from LCAs to complex numbers.
   * Fundamental methods: evaluation, composition, shift (translation), pullback, pushforward, point-wise operators (i.e. addition).

Example
^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from abelian import LCA, HomLCA, LCAFunc, voronoi
    from math import exp, pi
    # Create groups Z and R
    Z = LCA(orders = [0], discrete = [True])
    R = LCA(orders = [0], discrete = [False])

    # Create the Gaussian function on R^2
    function = LCAFunc(lambda x: exp(-pi*sum(j**2 for j in x)), domain = R**2)

    # Create an orthogonal sampling homomorphism
    phi = HomLCA([[0.05, 0], [0, 0.05]], source = Z**2, target = R**2)
    function_sampled = function.pullback(phi)

    # Approximate the two-dimensional integral of the gaussian
    scaling_factor = phi.A.det()
    integral_value = 0
    for element in phi.source.elements_by_maxnorm(list(range(50))):
        integral_value += function_sampled(element)
    print(integral_value * scaling_factor) # 0.999999998926396

    # Sample, periodize and take DFT of the Gaussian
    phi_p = HomLCA([[15, 0], [0, 15]], source = Z**2, target = Z**2)
    periodized = function_sampled.pushforward(phi_p.cokernel())
    dual_func = periodized.dft()
    DFT_ouput = dual_func.table * phi_p.A.det() * scaling_factor

    # Interpret the output of the DFT on R^2
    phi_periodize_ann = phi_p.annihilator()

    # Compute a Voronoi transversal function, interpret on R**2
    sigma = voronoi(phi.dual(), norm_p=2)
    for element in dual_func.domain.elements_by_maxnorm():
        value = dual_func(element)
        coords_on_R = sigma(phi_periodize_ann(element))

        # The function is invariant under Fourier transformation, so we can
        # compare the error analytically
        true_val = function(coords_on_R) # The function is invariant under FT
        approximated_val = abs(value)
        assert abs(true_val - approximated_val) < 0.01

Please see `the documentation <http://abelian.readthedocs.io/en/latest/>`_ for more examples and information.