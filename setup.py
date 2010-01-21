#!/usr/bin/env python

import setup_env

from distutils.core import (
    setup,
    Extension,
)

setup(
    name="pymat2",
    version=setup_env.VERSION,
    description="Python to Matlab interface",
    author="Ilya O.",
    author_email="vrghost@gmail.com",
    url="http://code.google.com/p/pymat2/",
    packages=["pymat2"],
    ext_modules=[
        Extension("_pymat2",
            sources=setup_env.compilable_sources,
        ),
    ],
)
