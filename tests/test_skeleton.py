#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from abelian.skeleton import fib

__author__ = "tommyod"
__copyright__ = "tommyod"
__license__ = "gpl3"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
