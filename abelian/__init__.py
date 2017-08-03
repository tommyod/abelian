#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pkg_resources

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = 'unknown'


from abelian.function import Function
from abelian.group import Group
from abelian.linalg.factorizations import smith_normal_form, hermite_normal_form


__all__ = ['identity', 'Function', 'Group']
