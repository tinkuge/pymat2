#!/usr/bin/env python

import setup_env

from distutils.core import (
    setup,
    Extension,
)

setup(
    name="pymat2",
    version=setup_env.version,
    description="Python to Matlab interface",
    author="Ilya O.",
    author_email="vrghost@gmail.com",
    url="http://code.google.com/p/pymat2/",
    packages=["pymat2"],
    ext_modules=[
        Extension("pymat2._pymat2",
            sources=setup_env.compilable_sources,
            include_dirs=[
                setup_env.MATLAB_INCLUDE_DIR,
            ],
            library_dirs=[
                setup_env.MATLAB_LIB_DIR,
            ],
            define_macros=[
                ("PYMAT_VERSION", '"%s"' % setup_env.version),
            ],
            extra_link_args=["-lmx", "-leng"],
        ),
    ],
)
