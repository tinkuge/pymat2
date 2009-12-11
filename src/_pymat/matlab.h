#ifndef __PYMAT_MATLAB_INCLUDED__
#define __PYMAT_MATLAB_INCLUDED__

#include "pymat_head.h"

#define MATLAB_OUTPUT_BUFFER_LEN  1024	/* 1K. Is that too little for you? */

void init_pymat_matlab(PyObject *module);

#endif /* __PYMAT_MATLAB_INCLUDED__ */