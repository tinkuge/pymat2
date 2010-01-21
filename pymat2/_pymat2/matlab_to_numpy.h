#ifndef __MATLAB_TO_NUPY_INCLUDED__
#define __MATLAB_TO_NUPY_INCLUDED__

#include "pymat_head.h"
#include <matrix.h>

PyObject* mx2char(const mxArray *);
PyObject* mx2numeric(const mxArray *);

#endif /* __MATLAB_TO_NUPY_INCLUDED__ */
