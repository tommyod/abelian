=======
abelian
=======

.. image:: https://readthedocs.org/projects/abelian/badge/?version=latest
   :target: http://abelian.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

``abelian`` is a Python library for computations on elementary locally compact abelian groups (LCAs).
The elementary LCAs are the groups R, Z, T = R/Z, Z_n and direct sums of these.
The Fourier transformation is defined on these groups.
With ``abelian`` it is possible to sample, periodize and perform Fourier
analysis on elementary LCAs using homomorphisms between groups.

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

.. image:: http://tommyodland.com/abelian/fourier_hexa_25.png

We create a Gaussian on R^2 and a homomorphism for sampling.

.. code:: python

    from abelian import LCA, HomLCA, LCAFunc, voronoi
    from math import exp, pi, sqrt
    Z = LCA(orders = [0], discrete = [True])
    R = LCA(orders = [0], discrete = [False])

    # Create the Gaussian function on R^2
    function = LCAFunc(lambda x: exp(-pi*sum(j**2 for j in x)), domain = R**2)

    # Create an hexagonal sampling homomorphism (lattice on R^2)
    phi = HomLCA([[1, 1/2], [0, sqrt(3)/2]], source = Z**2, target = R**2)
    phi = phi * (1/7) # Downcale the hexagon
    function_sampled = function.pullback(phi)

Next we approximate the two-dimensional integral of the Gaussian.

.. code:: python

    # Approximate the two dimensional integral of the Gaussian
    scaling_factor = phi.A.det()
    integral_sum = 0
    for element in phi.source.elements_by_maxnorm(list(range(20))):
        integral_sum += function_sampled(element)
    print(integral_sum * scaling_factor) # 0.999999997457763


We use the FFT to move approximate the Fourier transform of the Gaussian.

.. code:: python

    # Sample, periodize and take DFT of the Gaussian
    phi_p = HomLCA([[10, 0], [0, 10]], source = Z**2, target = Z**2)
    periodized = function_sampled.pushforward(phi_p.cokernel())
    dual_func = periodized.dft()

    # Interpret the output of the DFT on R^2
    phi_periodize_ann = phi_p.annihilator()

    # Compute a Voronoi transversal function, interpret on R^2
    sigma = voronoi(phi.dual(), norm_p=2)
    factor = phi_p.A.det() * scaling_factor
    total_error = 0
    for element in dual_func.domain.elements_by_maxnorm():
        value = dual_func(element)
        coords_on_R = sigma(phi_periodize_ann(element))

        # The Gaussian is invariant under Fourier transformation, so we can
        # compare the error using the analytical expression
        true_val = function(coords_on_R)
        approximated_val = abs(value)
        total_error += abs(true_val - approximated_val*factor)

    assert total_error < 10e-15

Please see `the documentation <http://abelian.readthedocs.io/en/latest/>`_ for more examples and information.