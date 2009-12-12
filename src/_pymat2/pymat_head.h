/* Global pymat header includes, system-dependant defines, etc */
#ifndef __PYMAT_HEAD_INCLUDED__
#define __PYMAT_HEAD_INCLUDED__

// We're not building a MEX file, we're building a standalone app.
#undef MATLAB_MEX_FILE
#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#if defined(NO_IMPORT_ARRAY) || defined(NO_IMPORT)
#	error No Numpy array allowed
#endif

#ifdef WIN32
#include <windows.h>
#endif

#ifndef max
#define max(x,y) ((x) > (y) ? (x) : (y))
#endif
#ifndef min
#define min(x,y) ((x) < (y) ? (x) : (y))
#endif

#define PY_MODULE_NAME	"_pymat2"
#define PYMAT_VERSION "0.0.2"

#define _STR(x) #x
#define STR(x) _STR(x)


/* 
* I know that it is evil, but it is simpliest of all hacks
* I can think of now.
* 
* If you have any proposals on how to remove this thing,
* notify me please.
*/
static int pyarray_works(void){
	int is_ok;
	if(!PyArray_API){
		is_ok = (_import_array() >= 0);
	}else{
		is_ok = 1;
	}
	return is_ok;
}

#endif /* __PYMAT_HEAD_INCLUDED__ */
