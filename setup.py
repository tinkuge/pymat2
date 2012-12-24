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
                setup_env.numpy_include,
            ],
            library_dirs=setup_env.MATLAB_LIB_DIRS,
            define_macros=[
                ("DISTUTILS_PYMAT_VERSION", '%s' % setup_env.version),
            ] + setup_env.extra_defines,
            extra_link_args=setup_env.extra_link_args,
            extra_compile_args=setup_env.extra_compile_args,
        ),
    ],
)
