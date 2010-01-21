#!/usr/bin/env python

import os

# The following variables are expected to be set by user

MATLAB_DIR = r"/usr/share/matlab"
MATLAB_INCLUDE_DIR = os.path.join(MATLAB_DIR, "extern", "include")
MATLAB_LIB_DIR = os.path.join(MATLAB_DIR, "extern", "lib", "glnxa64")


# The following code is not expected to be set by user

import glob

__this_dir__ = os.path.abspath(os.path.dirname(__file__))

version = file(os.path.join(__this_dir__, "pymat2.version")).read().strip()
compilable_source_dir = os.path.join(__this_dir__, "pymat2", "_pymat2")
compilable_sources = glob.glob(os.path.join(compilable_source_dir, "*.cpp"))

