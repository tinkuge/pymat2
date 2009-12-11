#include "numpy_to_matlab.h"
#include "exceptions.h"


template <class T>
void copyNumeric2Mx(T *pSrc, int pRows, int pCols, double *pDst, int *pStrides)
{
    int lRowDelta = pStrides[1]/sizeof(T);
    int lColDelta = pStrides[0]/sizeof(T);

    for (int lCol=0; lCol < pCols; lCol++) {
        T *lSrc = pSrc + lCol*lRowDelta;
        for (int lRow=0; lRow < pRows; lRow++, lSrc += lColDelta) {
            *pDst++ = *lSrc;
        }
    }
}

template <class T>
void copyCplxNumeric2Mx(T *pSrc, int pRows, int pCols, double *pRData, double *pIData, int *pStrides)
{
    int lRowDelta = pStrides[1]/sizeof(T);
    int lColDelta = pStrides[0]/sizeof(T);

    for (int lCol=0; lCol < pCols; lCol++) {
        T *lSrc = pSrc + lCol*lRowDelta;
        for (int lRow=0; lRow < pRows; lRow++, lSrc += lColDelta) {
            *pRData++ = lSrc[0];
            *pIData++ = lSrc[1];
        }
    }
}

static mxArray *makeMxFromNumeric(const PyArrayObject *pSrc)
{
    int lRows, lCols;
    bool lIsComplex;
    double *lR = 0;
    double *lI = 0;
    mxArray *lRetval = 0;

	if(pSrc->nd > 2){
		not_implemented_msg("Only 1-D and 2-D arrays are currently supported");
		return NULL;
	}

    lRows = (int)pSrc->dimensions[0]; /* dimensionality checked previously */
    if (pSrc->nd < 2) {
        lCols = 1;
    } else {
        lCols = (int)pSrc->dimensions[1]; /* dimensionality checked previously */
    }

    switch (pSrc->descr->type_num) {
    case PyArray_OBJECT:
        raise_pymat_error(PYMAT_ERR_NUMPY, "Non-numeric array types not supported");
        return NULL;
        
    case PyArray_CFLOAT:
    case PyArray_CDOUBLE:
        lIsComplex = true;
        break;
        
    default:
        lIsComplex = false;
    }
    
    lRetval = mxCreateDoubleMatrix(lRows, lCols, lIsComplex ? mxCOMPLEX : mxREAL);

	if (!lRetval) {
		return NULL;
	}

    lR = mxGetPr(lRetval);
    lI = mxGetPi(lRetval);

    switch (pSrc->descr->type_num) {
    case PyArray_CHAR:
        copyNumeric2Mx((char *)(pSrc->data), lRows, lCols, lR, pSrc->strides);
        break;

    case PyArray_UBYTE:
        copyNumeric2Mx((unsigned char *)(pSrc->data), lRows, lCols, lR, pSrc->strides);
        break;

    case PyArray_BYTE:
        copyNumeric2Mx((signed char *)(pSrc->data), lRows, lCols, lR, pSrc->strides);
        break;

    case PyArray_SHORT: 
        copyNumeric2Mx((short *)(pSrc->data), lRows, lCols, lR, pSrc->strides);
        break;

    case PyArray_INT:
        copyNumeric2Mx((int *)(pSrc->data), lRows, lCols, lR, pSrc->strides);
        break;

    case PyArray_LONG:
        copyNumeric2Mx((long *)(pSrc->data), lRows, lCols, lR, pSrc->strides);
        break;

    case PyArray_FLOAT:
        copyNumeric2Mx((float *)(pSrc->data), lRows, lCols, lR, pSrc->strides);
        break;

    case PyArray_DOUBLE:
        copyNumeric2Mx((double *)(pSrc->data), lRows, lCols, lR, pSrc->strides);
        break;

    case PyArray_CFLOAT:
        copyCplxNumeric2Mx((float *)(pSrc->data), lRows, lCols, lR, lI, pSrc->strides);
        break;

    case PyArray_CDOUBLE:
        copyCplxNumeric2Mx((double *)(pSrc->data), lRows, lCols, lR, lI, pSrc->strides);
        break;
    }

    return lRetval;

}

static mxArray *makeMxFromSeq(const PyObject *pSrc)
{
    mxArray *lRetval = 0;
    int i, lSize;

    PyArrayObject *lArray = (PyArrayObject *)PyArray_ContiguousFromObject(const_cast<PyObject *>(pSrc), PyArray_CDOUBLE, 0, 0);
	if (!lArray) {
		return NULL;
	}

    // If all imaginary components are 0, this is not a complex array.
    lSize = (int)PyArray_SIZE(lArray);
    const double *lPtr = (const double *)(lArray->data) + 1;  // Get at first imaginary element
    for (i=0; i < lSize; i++, lPtr += 2) {
        if (*lPtr != 0.0) break;
    }
    if (i >= lSize) {
        PyArrayObject *lNew = (PyArrayObject *)PyArray_Cast(lArray, PyArray_DOUBLE);
        Py_DECREF(lArray);
        lArray = lNew;
    }

    lRetval = makeMxFromNumeric(lArray);
    Py_DECREF(lArray);

    return lRetval;
}

mxArray *char2mx(const PyObject *pSrc)
{
    mxArray *lDst = 0;

    lDst = mxCreateString(PyString_AsString(const_cast<PyObject *>(pSrc)));
    if (!lDst) {
        raise_pymat_error(PYMAT_ERR_MATLAB_TO_NUMPY, "Unable to extract MATLAB string");
        return NULL;
    }
    return lDst;
}


mxArray* numeric2mx(const PyObject *pSrc)
{
    mxArray* lDst = NULL;



	if(!pyarray_works()){
		raise_pymat_error_with_value(
			PYMAT_ERR_NUMPY, 
			"Unable to perform this function without NumPy installed",
			(int)PyArray_API
		);
		return NULL;
	}

    if (PyArray_Check(pSrc)) {
        lDst = makeMxFromNumeric((const PyArrayObject *)pSrc);
    } else {
        lDst = makeMxFromSeq(pSrc);
    }

    return lDst;
}
