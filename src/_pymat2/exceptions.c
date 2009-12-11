/* Pymat exceptions classes and related functionality */

#include "exceptions.h"

/* XXX: Message order in this array MUST follow order of enums in exceptions.h*/
char *pymat_error_strings[] = {
	"PYMAT_ERR_UNKNOWN",
	"PYMAT_ERR_WRONG_INIT_ARGUMENT",
	"PYMAT_ERR_MATLAB_ENGINE_START",
	"PYMAT_ERR_MATLAB_ENGINE_STARTED",
	"PYMAT_ERR_MATLAB_ENGINE_NOT_STARTED",
	"PYMAT_ERR_MATLAB_ENGINE_STOP",
	"PYMAT_ERR_FUNCTION_ARGS",
	"PYMAT_ERR_EVAL_ERROR",
	"PYMAT_ERR_GET_ERROR",
	"PYMAT_ERR_MATLAB_TO_NUMPY",
	"PYMAT_ERR_MATLAB_MEMORY",
	"PYMAT_ERR_NUMPY",
	"PYMAT_ERR_MATLAB",
};
#define PYMAT_ERROR_COUNT (sizeof(pymat_error_strings) / sizeof(*pymat_error_strings))

int get_pymat_error_count(void){
	return PYMAT_ERROR_COUNT;
}

const char* get_pymat_error_string_for_code(const int code){
	return pymat_error_strings[code];
}

PyObject* raise_pymat_error_with_value(const int errorId, const char* msg, const int err_addinfo) {
	PyObject *err;

	err = PyTuple_Pack(3,
		PyInt_FromLong(errorId),
		PyString_FromString(msg),
		PyInt_FromLong(err_addinfo)
		);
	if (err != NULL) PyErr_SetObject(PymatError, err);
	Py_XDECREF(err);
	return NULL;
}

PyObject* raise_pymat_error(const int errorId, const char* msg) {
	PyObject *err;

	err = PyTuple_Pack(2,
		PyInt_FromLong(errorId),
		PyString_FromString(msg)
		);
	if (err != NULL) PyErr_SetObject(PymatError, err);
	Py_XDECREF(err);
	return NULL;
}

/* raise NotImplementedError; */
PyObject* not_implemented(void) {
	PyErr_SetNone(PyExc_NotImplementedError);
    return NULL;
}

void init_pymat_exceptions(PyObject *module){
	/* Add error code constants to module */
	for(int ii=0; ii < get_pymat_error_count(); ii++){
		PyModule_AddIntConstant(module, get_pymat_error_string_for_code(ii), ii);
	}

	/* Add pymat exception object */
	PymatError = PyErr_NewException(PY_MODULE_NAME ".PymatError", NULL, NULL);
	if (PymatError == NULL) return;
	Py_INCREF(PymatError);
	PyModule_AddObject(module, "PymatError", PymatError);
}

PyObject* not_implemented_msg(const char* msg) {
	PyErr_SetString(
		PyExc_NotImplementedError, msg);
    return NULL;
}