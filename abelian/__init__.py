#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pkg_resources

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = 'unknown'


from abelian.linalg.solvers import solve
from abelian.linalg.factorizations import smith_normal_form, hermite_normal_form
from abelian.groups import LCA
from abelian.morphisms import HomFGA, HomLCA, Homomorphism


__all__ = []

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
