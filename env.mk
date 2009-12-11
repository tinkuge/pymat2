OS := $(shell eval 'uname | tr "[:upper:]" "[:lower:]" | tr -c -d "[:lower:]"')
PYTHON ?= python

ifeq ($(OS),cygwinnt)
# assume running Windows

OBJ_EXT=obj
MSVC_ROOT_DIR := C:\Program Files\VisualStudio2008
PSDK_ROOT_DIR := C:\Program Files\Microsoft SDKs\Windows\v5.0
PYTHON_ROOT_DIR := C:\Python26
MATLAB_EXTERN_DIR := C:\programming\src\external\Matlab
NUMPY_DIR = ${PYTHON_ROOT_DIR}\Lib\site-packages\numpy\core

# Compiler settings follow

INCLUDES = \
	/I "$(PSDK_ROOT_DIR)\include" \
	/I "$(MSVC_ROOT_DIR)\VC\include" \
	/I "$(MATLAB_EXTERN_DIR)\include" \
	/I "$(PYTHON_ROOT_DIR)\include" \
	/I "$(NUMPY_DIR)\include"


CFLAGS = /O2 /Ob2 /Ot \
	 /MD /Gd \
	 /GF /FD /EHsc /MD /W3 /WX /nologo /c \
	 /Gd /TP

# Linker settings follow

LIB_DIRS = \
    /LIBPATH:"$(PSDK_ROOT_DIR)\\Lib" \
	/LIBPATH:"$(MSVC_ROOT_DIR)\\VC\\lib" \
	/LIBPATH:"$(MATLAB_EXTERN_DIR)\\lib\\win32\\microsoft\\msvc70" \
	/LIBPATH:"$(PYTHON_ROOT_DIR)\\libs" \

LINK_OPTS = \
	 /DLL /MANIFEST:NO /SUBSYSTEM:WINDOWS \
	 /OPT:NOREF /DYNAMICBASE:NO /NOLOGO

EXTRA_LIBS = libeng.lib libmx.lib

# Build environement settings follow

CYGWIN_MSVC_ROOT = "$(shell cygpath "$(MSVC_ROOT_DIR)")"

CC = $(CYGWIN_MSVC_ROOT)/VC/bin/cl.exe $(CFLAGS) $(INCLUDES)
LINK = $(CYGWIN_MSVC_ROOT)/VC/bin/link.exe $(LINK_OPTS) $(LIB_DIRS) $(EXTRA_LIBS)

# XXX: Need mspdb80.dll in PATH
# For me there are several version on them, and one that is actually can be used
# is located in
# C:\Program Files\Microsoft Visual Studio 8\Common7\IDE

else
	# assume running kind of *nix
endif
