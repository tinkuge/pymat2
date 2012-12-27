#!/usr/bin/env python

from distutils.sysconfig import get_python_lib
import glob
import os
import platform

is_64_bit_os = "64 bit" in platform.python_compiler()
# The following variables are expected to be set by user


if os.name == "nt":
    MATLAB_DIR = r"C:\Program Files (x86)\MATLAB\R2009b"
    lib_dir =os.path.join(MATLAB_DIR, "extern", "lib")

    if is_64_bit_os:
        lib_dir = os.path.join(lib_dir, "win64")
    else:
        lib_dir = os.path.join(lib_dir, "win32")

    MATLAB_LIB_DIRS = [os.path.join(lib_dir, "microsoft")]
elif os.name == "mac":
    raise NotImplementedError("Please, setup Matlab-related stuff here.")
else:
    MATLAB_DIR = r"/usr/share/Matlab"
    MATLAB_LIB_DIRS = [
        os.path.join(MATLAB_DIR, "extern", "lib", "glnxa64"),
        os.path.join(MATLAB_DIR, "bin", "glnxa64"),
    ]

MATLAB_INCLUDE_DIR = os.path.join(MATLAB_DIR, "extern", "include")
if not os.path.exists(MATLAB_INCLUDE_DIR):
    raise RuntimeError("Matlab include directory not found.")

# The following code is not expected to be set by user


__this_dir__ = os.path.abspath(os.path.dirname(__file__))

version = file(os.path.join(__this_dir__, "pymat2.version")).read().strip()
compilable_source_dir = os.path.join(__this_dir__, "pymat2", "_pymat2")
compilable_sources = glob.glob(os.path.join(compilable_source_dir, "*.cpp"))

numpy_dir = os.path.join(get_python_lib(), "numpy")
numpy_include = os.path.join(numpy_dir, "core", "include")

if not os.path.exists(numpy_include):
    raise RuntimeError("Numpy include directory not found.")

if os.name == "nt":
    extra_link_args=["libeng.lib", "libmx.lib"]
else:
    extra_link_args=["-lmx", "-leng"]

if os.name == "nt":
    extra_defines=[("WINDOWS", "")]
else:
    extra_defines=[("NIX", "")]

extra_compile_args=[]
