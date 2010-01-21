#include "matlab.h"
#include "exceptions.h"

#include "matlab_to_numpy.h"
#include "numpy_to_matlab.h"

#include <engine.h>

PyDoc_STRVAR(MatlabObject_doc,
"Object representing single Matlab instance.\n\
\n\
Takes single argument (named 'startCmd') on *nix operationg systems,\n\
Does not take any arguments on Windows.\n\
");

/* 
   Dummy return status to be set in 'matlab return status'
   structure when engine is started.

   Actual value of constant determined by fair "random.randint(0, 2048)"
   call in Python interpreter.
*/
#define NO_MATLAB_RETURN_STATUS		1946
typedef struct {
	PyObject_HEAD;
	char *start_command;
	Engine *matlab_engine;
	int matlab_engine_return_status;

	char *matlab_engine_output_buffer;
	int matlab_engine_output_buffer_len;
} MatlabObject;
// Actual object function follow

#define _MATLAB_MUST_BE_RUNNING \
	if(!self->matlab_engine){\
		return raise_pymat_error(PYMAT_ERR_MATLAB_ENGINE_NOT_STARTED, "Matlab engine is not running.");\
	}


PyDoc_STRVAR(Matlab_put_array_doc, "\
void put(name, array) : Places a matrix into the MATLAB session\n\
\n\
This function places the given array into a MATLAB workspace under the\n\
name given with the 'name' parameter (which must be a string).\n\
\n\
The 'array' parameter must be either a NumPy array, list, or tuple\n\
containing real or complex numbers, or a string. The MATLAB array will\n\
have the same shape and same values, after conversion to double-precision\n\
floating point, if necessary. A string parameter is converted to a MATLAB\n\
char-valued array.\n\
");
PyObject* Matlab_put_array(MatlabObject *self, PyObject *args){
	char *lName;
 
    PyObject *lSource;
    mxArray *lArray = NULL;

	if (! PyArg_ParseTuple(args, "sO:put", &lName, &lSource)){
		return raise_pymat_error(PYMAT_ERR_FUNCTION_ARGS, "Invalid arguments supplied.");
	}
    Py_INCREF(lSource);

    if (PyString_Check(lSource)) {
        lArray = char2mx(lSource);
    } else {
        lArray = numeric2mx(lSource);
    }
    Py_DECREF(lSource);

    if (!lArray ) {
        Py_RETURN_NONE;
    }

    if (engPutVariable(self->matlab_engine, lName, lArray)) {
        return raise_pymat_error(
			PYMAT_ERR_MATLAB, "Unable to put matrix into MATLAB workspace");
    }

    mxDestroyArray(lArray);

    Py_INCREF(Py_None);
    return Py_None;

}


PyDoc_STRVAR(Matlab_get_array_doc, "\
get(name) : Gets a matrix from the MATLAB session\n\
\n\
This function extracts the matrix with the given name from a MATLAB.\n\
The name parameter must be a string describing\n\
the name of a matrix in the MATLAB workspace. On Win32 platforms,\n\
only double-precision floating point arrays (real or complex) are supported.\n\
1-D character strings are supported on UNIX platforms.\n\
\n\
Only 1-dimensional and 2-dimensional arrays are currently supported. Cell\n\
arrays, structure arrays, etc. are not yet supported.\n\
\n\
The return value is a NumPy array with the same shape and elements as the\n\
MATLAB array.\n\
");
PyObject * Matlab_get_array(MatlabObject *self, PyObject *args){
   char *lName;

 
   mxArray *lArray = 0;
   PyObject *rv = NULL;

   _MATLAB_MUST_BE_RUNNING;

	if (! PyArg_ParseTuple(args, "s:get", &lName)) {
		  return raise_pymat_error(PYMAT_ERR_FUNCTION_ARGS, "Invalid arguments supplied.");
	}

   lArray = engGetVariable(self->matlab_engine, lName);
   if (!lArray) {
      return raise_pymat_error(PYMAT_ERR_GET_ERROR, "Unable to get matrix from MATLAB workspace");
   }	

   if (mxIsChar(lArray)) {
      rv = (PyObject *)mx2char(lArray);
   } else if (mxIsDouble(lArray)) {
      rv = (PyObject *)mx2numeric(lArray);
   } else {
      rv = raise_pymat_error(
		  PYMAT_ERR_MATLAB_TO_NUMPY, "Only floating-point and character arrays are currently supported");
   }
   mxDestroyArray(lArray);

   return rv;
}


static PyObject *Matlab_eval(MatlabObject *self, PyObject *args, PyObject *kwds){
	char *command=NULL;
	int rc, eval_rc;
	
	_MATLAB_MUST_BE_RUNNING;

	rc = PyArg_ParseTuple(args, "z", &command);
	if(!rc){
		return raise_pymat_error(PYMAT_ERR_FUNCTION_ARGS, "Failed to parse command.");
	};
	eval_rc = engEvalString(self->matlab_engine, command);
	if(eval_rc){
		return raise_pymat_error_with_value(
			PYMAT_ERR_EVAL_ERROR, "Failed to execute command.", eval_rc);
	}
	if(self->matlab_engine_output_buffer){
		return PyString_FromString(self->matlab_engine_output_buffer);
	}else{
		Py_RETURN_NONE;
	}
}

PyDoc_STRVAR(Matlab_set_output_buffer_size_doc, 
	"Set matlab output buffer size. (0 to turn off, " STR(MATLAB_OUTPUT_BUFFER_LEN) " by default)");
static PyObject *Matlab_set_output_buffer_size(MatlabObject *self, PyObject *args, PyObject *kwds){
	int desired_size=0, rc;
	
	_MATLAB_MUST_BE_RUNNING;

	rc = PyArg_ParseTuple(args, "i", &desired_size);
	if(!rc || desired_size < 0){
		return raise_pymat_error(PYMAT_ERR_FUNCTION_ARGS, "Failed to parse buffer size.");
	};

	if(self->matlab_engine_output_buffer){
		PyMem_Free(self->matlab_engine_output_buffer);
	}

	self->matlab_engine_output_buffer_len = desired_size;
	if(desired_size){
		self->matlab_engine_output_buffer = (char*)PyMem_Malloc(desired_size + 1);
	}else{
		self->matlab_engine_output_buffer = NULL;
	}
	engOutputBuffer(self->matlab_engine, 
		self->matlab_engine_output_buffer, self->matlab_engine_output_buffer_len);
	Py_RETURN_NONE;
}

PyDoc_STRVAR(Matlab_stop_doc, 
"Start Matlab engine process.");
static PyObject *Matlab_stop(MatlabObject *self){
	_MATLAB_MUST_BE_RUNNING;
	if(engClose(self->matlab_engine)){
		return raise_pymat_error(PYMAT_ERR_MATLAB_ENGINE_STOP, "Matlab engine closing error.");
	}
#if WIN32
	assert(NO_MATLAB_RETURN_STATUS > 0);
	while(self->matlab_engine_return_status == NO_MATLAB_RETURN_STATUS){
		/* In Windows "Sleep" has delay parameter measured in msec. */
		Sleep(100);
	}
#endif
	self->matlab_engine = NULL;
	Py_RETURN_NONE;
}

PyDoc_STRVAR(Matlab_start_doc, 
"Start Matlab engine process.");
static PyObject *Matlab_start(MatlabObject *self){
	if(self->matlab_engine){
		return raise_pymat_error(PYMAT_ERR_MATLAB_ENGINE_STARTED, "Matlab engine is already started.");
	}
	/* 
	   Creating separate MATLAB instance for each process
	   is safer.

	   *nix users are out of luck, as usual.
	 */
#ifdef WIN32
	self->matlab_engine = engOpenSingleUse(
		self->start_command, NULL, 
		&(self->matlab_engine_return_status)
	);
	self->matlab_engine_return_status = NO_MATLAB_RETURN_STATUS;
#else
	self->matlab_engine = engOpen(self->start_command);
#endif
	if(!self->matlab_engine){
		return raise_pymat_error(PYMAT_ERR_MATLAB_ENGINE_START, "Failed to start Matlab Engine");
	}
	// Engine started, register output buffer.
	if(self->matlab_engine_output_buffer){
		engOutputBuffer(self->matlab_engine, 
			self->matlab_engine_output_buffer, self->matlab_engine_output_buffer_len);
	}
	Py_RETURN_NONE;
}

PyDoc_STRVAR(Matlab_is_running_doc, 
"Return true if matlab instance is instantiated.");
static PyObject *Matlab_is_running(MatlabObject *self){
	if(self->matlab_engine){
		Py_RETURN_TRUE;
	}else{
		Py_RETURN_FALSE;
	}
}


// Service functions follow
static void Matlab_dealloc(MatlabObject *self){
	if(self->matlab_engine_output_buffer){
		PyMem_Free(self->matlab_engine_output_buffer);
	}
	self->ob_type->tp_free((PyObject *)self);
}

static PyObject* Matlab_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    return type->tp_alloc(type, 0);
}

static int Matlab_init(MatlabObject *self, PyObject *args, PyObject *kwds) {
	char *kwlist[] = {(char*)"startCmd", NULL};
	int rc;

	char *startCmd = NULL;
	int startCmdLen = 0;

	rc = PyArg_ParseTupleAndKeywords(args, kwds, "|z#:startCmd", kwlist, &startCmd, &startCmdLen);
	if(!rc){ return -1; };
	
	self->start_command = NULL;
	self->matlab_engine = NULL;

	self->matlab_engine_output_buffer_len = MATLAB_OUTPUT_BUFFER_LEN;
	self->matlab_engine_output_buffer = (char*)PyMem_Malloc(MATLAB_OUTPUT_BUFFER_LEN + 1);

#ifdef WIN32
	if(startCmdLen){
		raise_pymat_error(
			PYMAT_ERR_WRONG_INIT_ARGUMENT, "Constructor arguments not supported on Windows.");
		return -1;
	}	
#else
	if(startCmd && startCmdLen){
		self->start_command = (char*)PyMem_Malloc(startCmdLen + 1);
		strncpy(self->start_command, startCmd, startCmdLen+1);
	}
#endif	

	return 0;
}

static PyMethodDef Matlab_methods[] = {
	{"start", (PyCFunction)Matlab_start, METH_NOARGS, Matlab_start_doc},
	{"stop", (PyCFunction)Matlab_stop, METH_NOARGS, Matlab_stop_doc},

	{"eval", (PyCFunction)Matlab_eval, METH_VARARGS, "Eval string in Matlab"},
	{"setOutputBufferSize", 
		(PyCFunction)Matlab_set_output_buffer_size, METH_VARARGS, Matlab_set_output_buffer_size_doc},
	{"getArray", (PyCFunction)Matlab_get_array, METH_VARARGS, Matlab_get_array_doc},
	{"putArray", (PyCFunction)Matlab_put_array, METH_VARARGS, Matlab_put_array_doc},
	{"isRunning", (PyCFunction)Matlab_is_running, METH_NOARGS, Matlab_is_running_doc},

	{NULL}
};

static PyMemberDef Matlab_members[] = {
	{(char*)"startCommand", T_STRING, offsetof(MatlabObject, start_command), 
		RO, (char*)"Start command used to start given Matlab instance"},
	{(char*)"outputBufferSize", T_INT, offsetof(MatlabObject, matlab_engine_output_buffer_len),
		RO, (char*)"Currently set output buffer size"},
	{(char*)"engineReturnStatus", T_INT, offsetof(MatlabObject, matlab_engine_return_status),
		RO, (char*)"Matlab object return status."},
	{NULL}
};

static PyTypeObject MatlabType = {
	PyObject_HEAD_INIT(NULL)
	0,
	PY_MODULE_NAME ".Matlab", /* tp_name */
	sizeof(MatlabObject), /* tp_basicsize */
	0, /*  tp_itemsize */
	(destructor)Matlab_dealloc, /* tp_dealloc */

	0, /* tp_print */
    0, /* tp_getattr */
	0, /* tp_setattr */
    0, /* tp_compare */
    0, /* tp_repr */
	0, /* *tp_as_number */
    0, /* *tp_as_sequence */
    0, /* *tp_as_mapping */
    0, /* tp_hash */
    0, /* tp_call */
    0, /* tp_str */
    0, /* tp_getattro */
    0, /* tp_setattro */
    0, /* *tp_as_buffer */
	
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
	MatlabObject_doc, /* tp_doc */
	
	0, /* tp_traverse */
    0, /* tp_clear */
	0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */

	Matlab_methods, /* tp_methods */
	Matlab_members, /* tp_members */
	
	0, /* *tp_getset */
    0, /* *tp_base */
    0, /* *tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */

	(initproc)Matlab_init, /* tp_init */
	0, /* tp_alloc */
	Matlab_new
};

void init_pymat_matlab(PyObject *module){
	if (PyType_Ready(&MatlabType) < 0) return;
	Py_INCREF(&MatlabType);
    PyModule_AddObject(module, "Matlab", (PyObject *)&MatlabType);
}
