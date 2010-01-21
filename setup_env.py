#!/usr/bin/env python

# The following variables are expected to be set by user

MATLAB_DIR = r"/usr/share/matlab"

# The following code is not expected to be set by user

import os

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
VERSION = file(os.path.join(THIS_DIR, "pymat2.version")).read().strip()

