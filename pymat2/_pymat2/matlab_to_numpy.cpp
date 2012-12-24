#include "matlab_to_numpy.h"

#include "exceptions.h"

PyObject* mx2char(const mxArray *pArray)
{
    const size_t buflen = mxGetM(pArray)*mxGetN(pArray) + 1;
    char *buf;
    PyObject *lRetval;

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

int go_next_coord(npy_intp* coords, PyArrayObject* controlled){
	int ii, n_dims, rv;

	n_dims = PyArray_NDIM(controlled);
	rv = 0;

	for(ii = 0; ii < n_dims; ii++){
		if(coords[ii] >= PyArray_DIM(controlled, ii) - 1){
			coords[ii] = 0;
		}
		else
		{
			coords[ii] ++;
			rv = 1;
			break;
		}
	}
	return rv;
}

void copy_mx_to_numpy(const mxArray *inArray, PyArrayObject* outArray){
	// XXX: array dimensions are not checked and are supposed to match
	void *item;
	int isComplex, ii;
	double *inData, *inImgData;
	npy_cdouble *valImag;
	npy_intp *coords;

	isComplex = mxIsComplex(inArray);
	inData = mxGetPr(inArray);

	coords = (npy_intp*)PyMem_Malloc(sizeof(npy_intp) * PyArray_NDIM(outArray));
	for(ii = 0; ii < PyArray_NDIM(outArray); ii++){
		coords[ii] = 0;
	}

	if(isComplex){
		inImgData = mxGetPi(inArray);
	}
	else
	{
		inImgData = NULL;
	}

	do
	{
		item = PyArray_GetPtr(outArray, coords);
		assert(item);
		if(isComplex)
		{
			valImag = (npy_cdouble*)item;
			valImag->real = (npy_double)(*inData++);
			valImag->imag = (npy_double)(*inImgData++);
		}
		else
		{
			*((npy_double*)item) = (npy_double)(*inData++);
		}
	}while(go_next_coord(coords, outArray));

	PyMem_Free(coords);
}

PyObject* mx2numeric(const mxArray *pArray)
{
    	mwSize number_of_matlab_dimensions;
	const mwSize *matlab_dimensions;

	size_t		number_of_python_dimensions;
	npy_intp	*python_dimensions;
	PyArrayObject* lRetval;

	int ii;

	if(!pyarray_works()){
		return raise_pymat_error(
			PYMAT_ERR_NUMPY, "Unable to perform this function without NumPy installed");
	}

	assert(mxIsDouble(pArray));

	// mxGetNumberOfDimensions always returns '2' or more
    number_of_matlab_dimensions = mxGetNumberOfDimensions(pArray);
    matlab_dimensions = mxGetDimensions(pArray);


	number_of_python_dimensions = number_of_matlab_dimensions;
	python_dimensions = (npy_intp*)PyMem_Malloc(sizeof(npy_int) * number_of_python_dimensions);
	for(ii=0; ii < number_of_python_dimensions; ii++){
		python_dimensions[ii] = (npy_int)matlab_dimensions[ii];
	}

    lRetval = (PyArrayObject *)PyArray_SimpleNew(
		number_of_python_dimensions, python_dimensions,
        mxIsComplex(pArray) ? PyArray_CDOUBLE : PyArray_DOUBLE
	);
	PyMem_Free(python_dimensions);

    if (!lRetval) Py_RETURN_NONE;

	copy_mx_to_numpy(pArray, lRetval);
    return (PyObject*)lRetval;
}
