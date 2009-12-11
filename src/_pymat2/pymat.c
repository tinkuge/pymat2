/*
    PYMAT -- A Python module that exposes the MATLAB engine interface
             and supports passing NumPy arrays back and forth.

  Revision History
  ----------------

  Version 1.0 -- December 26, 1998, Andrew Sterian (asterian@umich.edu)
     * Initial release
  Version 1.1 -- January 18, 2003, Axel Kowald (kowald@molgen.mpg.de)
     * Modified for Matlab 6.5 and Python 2.2
*/
#include "pymat_head.h"
#include "exceptions.h"
#include "matlab.h"

static PyMethodDef PymatMethods[] = {
	/* Yep, its empty. */
	{ 0}
};

PyMODINIT_FUNC init_pymat2(void)
{
    PyObject *module = Py_InitModule(PY_MODULE_NAME, PymatMethods);

	/* Add __version__ field to module */
	PyModule_AddStringConstant(module, "__version__", PYMAT_VERSION);
	init_pymat_exceptions(module);
	init_pymat_matlab(module);
}
