#include "matlab_to_numpy.h"
#include "numpy_to_matlab.h"
#include "exceptions.h"

PyObject* mx2char(const mxArray *pArray)
{
    int buflen;
    char *buf;
    PyObject *lRetval;

    buflen = mxGetM(pArray)*mxGetN(pArray) + 1;
    buf = (char *)mxCalloc(buflen, sizeof(char));
	if(!buf){
		return raise_pymat_error(
			PYMAT_ERR_MATLAB_MEMORY, "Out of MATLAB memory");
	}

    if (mxGetString(pArray, buf, buflen)) {
		mxFree(buf);
        return raise_pymat_error(
			PYMAT_ERR_MATLAB_TO_NUMPY, "Unable to extract MATLAB string");
	}

    lRetval = (PyObject *)PyString_FromString(buf);
    mxFree(buf);
    return lRetval;
}


PyObject* mx2numeric(const mxArray *pArray)
{
    int nd;
    const int *dims;
    int lMyDim;
    PyArrayObject *lRetval = 0;
    int lRows, lCols;
    const double *lPR;
    const double *lPI;

	if(!pyarray_works()){
		return raise_pymat_error(
			PYMAT_ERR_NUMPY, "Unable to perform this function without NumPy installed");
	}

    nd = mxGetNumberOfDimensions(pArray);
    if (nd > 2) {
        return not_implemented_msg(
			"Only 1-D and 2-D arrays are currently supported");
    }

    dims = mxGetDimensions(pArray);

    if (nd == 2 && dims[0] == 1 || dims[1] == 1) {
        // It's really 1-D
        lMyDim = max(dims[0], dims[1]);
        dims = & lMyDim;
        nd=1;
    }

    lRetval = (PyArrayObject *)PyArray_SimpleNew(
		nd, const_cast<int *>(dims), 
        mxIsComplex(pArray) ? PyArray_CDOUBLE : PyArray_DOUBLE
	);
    if (!lRetval) Py_RETURN_NONE;

    lRows = mxGetM(pArray);
    lCols = mxGetN(pArray);
    lPR = mxGetPr(pArray);
    if (mxIsComplex(pArray)) {
        lPI = mxGetPi(pArray);

        for (int lCol = 0; lCol < lCols; lCol++) {
            double *lDst = (double *)(lRetval->data) + 2*lCol;
            for (int lRow = 0; lRow < lRows; lRow++, lDst += 2*lCols) {
                lDst[0] = *lPR++;
                lDst[1] = *lPI++;
            }
        }
    } else {
        for (int lCol = 0; lCol < lCols; lCol++) {
            double *lDst = (double *)(lRetval->data) + lCol;
            for (int lRow = 0; lRow < lRows; lRow++, lDst += lCols) {
                *lDst = *lPR++;
            }
        }
    }

    return (PyObject*)lRetval;
}
