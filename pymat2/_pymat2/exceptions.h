/* Pymat exceptions classes and related functionality */
#ifndef __PYMAT_EXCEPTIONS_INCLUDED__
#define __PYMAT_EXCEPTIONS_INCLUDED__

#include "pymat_head.h"

static PyObject *PymatError;

/* List of possible errors */
enum { 
	PYMAT_ERR_UNKNOWN=0,
	PYMAT_ERR_WRONG_INIT_ARGUMENT,
	PYMAT_ERR_MATLAB_ENGINE_START,
	PYMAT_ERR_MATLAB_ENGINE_STARTED,
	PYMAT_ERR_MATLAB_ENGINE_NOT_STARTED,
	PYMAT_ERR_MATLAB_ENGINE_STOP,
	PYMAT_ERR_FUNCTION_ARGS,
	PYMAT_ERR_EVAL_ERROR,
	PYMAT_ERR_GET_ERROR,
	PYMAT_ERR_MATLAB_TO_NUMPY,
	PYMAT_ERR_MATLAB_MEMORY,
	PYMAT_ERR_NUMPY,
	PYMAT_ERR_MATLAB,
};

static PyObject* pymat_exception_dict;

PyObject* raise_pymat_error_with_value(const int, const char*, const int);
PyObject* raise_pymat_error(const int errorId, const char* msg);
PyObject* not_implemented(void);
PyObject* not_implemented_msg(const char* msg);
void init_pymat_exceptions(PyObject *module);

#endif /* __PYMAT_EXCEPTIONS_INCLUDED__ */
