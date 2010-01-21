#!/usr/bin/env python

from distutils.sysconfig import get_python_lib
import glob
import os

# The following variables are expected to be set by user

if os.name == "nt":
    MATLAB_DIR = r"C:\programming\src\external\Matlab"
    MATLAB_LIB_DIR = os.path.join(MATLAB_DIR, 
        "extern", "lib", "win32", "microsoft")
else:
    MATLAB_DIR = r"/usr/share/matlab"
    MATLAB_LIB_DIR = os.path.join(MATLAB_DIR, "extern", "lib", "glnxa64")

MATLAB_INCLUDE_DIR = os.path.join(MATLAB_DIR, "extern", "include")


# The following code is not expected to be set by user


__this_dir__ = os.path.abspath(os.path.dirname(__file__))

version = file(os.path.join(__this_dir__, "pymat2.version")).read().strip()
compilable_source_dir = os.path.join(__this_dir__, "pymat2", "_pymat2")
compilable_sources = glob.glob(os.path.join(compilable_source_dir, "*.cpp"))

numpy_dir = os.path.join(get_python_lib(), "numpy")
numpy_include = os.path.join(numpy_dir, "core", "include")

if os.name == "nt":
    extra_link_args=["libeng.lib", "libmx.lib"]
else:
    extra_link_args=["-lmx", "-leng"]
