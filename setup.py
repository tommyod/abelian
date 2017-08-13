#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for abelian.

    This file was generated with PyScaffold 2.5.7, a tool that easily
    puts up a scaffold for your new Python project. Learn more under:
    http://pyscaffold.readthedocs.org/
"""

import sys
from setuptools import setup, find_packages
from abelian import __version__
VERSION = __version__

with open('README.rst', 'r') as file:
    README_TEXT = file.read()


def setup_package():
    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    setup(setup_requires=['six', 'pyscaffold>=2.5a0,<2.6a0'] + sphinx,
          use_pyscaffold=False,
          name = 'abelian',
          description="Computations on abelian groups.",
          long_description = README_TEXT,
          version=VERSION,
          packages=find_packages(),
          author = "Tommy Odland",
          author_email= 'tommy.odland@student.uib.no',
          url="https://github.com/tommyod/abelian/")


if __name__ == "__main__":
    setup_package()
