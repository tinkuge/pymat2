#ifndef __NUMPY_TO_MATLAB_INCLUDED__
#define __NUMPY_TO_MATLAB_INCLUDED__

#include "pymat_head.h"
#include <matrix.h>

mxArray* char2mx(const PyObject *pSrc);
mxArray* numeric2mx(const PyObject *pSrc);

#endif /*__NUMPY_TO_MATLAB_INCLUDED__*/