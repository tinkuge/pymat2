#!/usr/bin/env python

import os

THIS_DIR = os.path.abspath(os.path.dirname(__file__))

from distutils.core import (
    setup,
    Extension,
)

from pymat2 import _pymat2

setup(
    name="pymat2",
    version=_pymat2.__version__,
    description="Python to Matlab interface",
    author="Ilya O.",
    author_email="vrghost@gmail.com",
    url="http://code.google.com/p/pymat2/",
    packages=["pymat2"],
    package_data={"pymat2": ["_pymat2.pyd"]},
)
