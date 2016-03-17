

# Running #

## Windows x64 ##

The catch with 64-bit distribution of Windows is that:
  1. It is impossible to call 64-bit code from 32-bit application
  1. Matlab installer detects x64 host and installs 64-bit Matlab
  1. Therefore, 32-bit distribution of Python and Pymat2 are unable to call x64 Matlab

Therefore, to be able to run x64 Matlab you will need:
  1. 64-bit distribution of Python (available from [official site](http://www.python.org/getit/))
  1. 64-bit distribution of NumPy (required by pymat2)
    * As of December 2012 the [official site](http://www.scipy.org/Download#head-f64942d62faddeb27278a2c735e81ef2a7349db0) does not provide official Windows x64 builds, but it does point where unofficial installer can be [downloaded](http://www.lfd.uci.edu/~gohlke/pythonlibs/) from.

## DLL import errors ##

Sometimes, after fresh install calling `import pymat2` fails with following error:
```
Python 2.7.3 (default, Apr 10 2012, 23:24:47) [MSC v.1500 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import pymat2
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Python27_x64\lib\site-packages\pymat2\__init__.py", line 1, in <module>
    from .Matlab import Matlab
  File "C:\Python27_x64\lib\site-packages\pymat2\Matlab.py", line 7, in <module>

    from . import (
ImportError: DLL load failed: The specified module could not be found.
```

This is caused by Pymat failing to locate DLLs that it depends upon. This problem is resolved by adding directory containing missing DLLs to system `%PATH%`.

  * Most frequent cause for this error is lack of Matlabs' `LIBENG.DLL` and `LIBMX.DLL` in the `%PATH%` that pymat **require** to operate. These DLLs are located in place like `C:\Program Files\MATLAB\R2009b\bin\win64`. Locate the directory containing them and manually add it to the `PATH`.
  * pymat depends on hundreds of DLLs (mostly these are optional recursive dependencies -- dependencies that pymats' dependencies depend upon). If you are experiencing problems, please download [Dependency Walker](http://www.dependencywalker.com/) program and see what DLLs are missing from `%PATH%`.

# Compiling #

The compilation process is invoked via calling `python.exe setup.py %action%` in the root of Pymat2 checkout directory.
`%action%` can be:
  * `bdist` -- creates zip-file with binary distribution
  * `bdist_wininst` -- creates windows installer for pymat2 module
  * `install` -- installs directly in to the Python distribution that is executing `setup.py`.

For more detail on how to work with `setup.py` please refer to the [official Python documentation](http://docs.python.org/2/distutils/builtdist.html).

## Windows x64 ##

As of end of 2012, Python 2.x is compiled by the MSVC 9.0 compiler (part of Visual Studio 2008). But back in 2008 Microsoft had decided to ship its Visual Studio with only x86 compiler enabled by default.

### Visual Studio 2008 Professional Edition ###

Legends are that "Professional" edition of VS2008 do feature an optional x64 compiler that can be selected for installation ether in the process of installation or in the "Modify setup" dialogue (called from "Add/Remove programs") for already existing install.

Sadly, I do not own a license for this edition of Visual Studio, so I can not give any hints on how to fix any problems that might arise.

### Visual Studio 2008 Express Edition ###

Here are steps that I had to make to compile x64 build of Pymat2 with VS2008 Express Edition.

  1. Install The Visual Studio 2008 Express Edition (available [here](http://www.microsoft.com/en-us/download/details.aspx?id=6506))
  1. Then, follow instructions provided in the ["How can I set up Microsoft Visual Studio 2008 Express Edition for use with MATLAB 7.7 (R2008b) on 64-bit Windows?"](http://www.mathworks.co.uk/support/solutions/en/data/1-6IJJ3L/) topic to get actual x64 compiler:
    1. [Download the Windows Server 2008 & .NET 3.5 SDK](http://www.microsoft.com/downloads/details.aspx?FamilyId=F26B1AA4-741A-433A-9BE5-FA919850BDBF&displaylang=en)
    1. Check that "Developer Tools"->"Visual C++ Compilers" on the "Installation Options" screen is checked
    1. Verify that x64 components were indeed installed by checking for presence `amd64` directory in the `%INSTALLROOT%\Microsoft Visual Studio 9.0\VC\bin` directory and that `amd64` directory contains executable files like `cl.exe` and `link.exe`
  1. Verify that `vcvarsall.bat` (located in `%INSTALLROOT%\Microsoft Visual Studio 9.0\VC`) initializes correctly for x64 architecture:
    1. Open command prompt, `CD` to the `%INSTALLROOT%\Microsoft Visual Studio 9.0\VC`
    1. Execute `vcvarsall.bat x64`
    1. Expected **correct** output is something similar to this:
```
C:\Users\Ilya>cd /d "C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC"

C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC>vcvarsall.bat x64
Setting environment for using Microsoft Visual Studio 2008 Beta2 x64 tools.

C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC>
```
    1. However, it is possible that it will be something like this:
```
C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC>vcvarsall.bat x64
The specified configuration type is missing.  The tools for the
configuration might not be installed.

C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC>
```
> > In this case:
      1. Open `vcvarsall.bat` in a text editor
      1. Search for (lines 18--21 in my case)
```
:amd64
if not exist "%~dp0bin\amd64\vcvarsamd64.bat" goto missing
call "%~dp0bin\amd64\vcvarsamd64.bat"
goto :eof
```
      1. Replace all occurrences of string `"%~dp0bin\amd64\vcvarsamd64.bat"` with `"%~dp0bin\vcvars64.bat"`
        * If you prefer to have diff of original and resulting version, here it is:
```
--- vcvarsall.bat       2012-12-25 12:11:48.616922800 +0000
+++ vcvarsall.bat       2012-12-24 19:38:01.068763300 +0000
@@ -16,8 +16,9 @@
 goto :eof

 :amd64
-if not exist "%~dp0bin\amd64\vcvarsamd64.bat" goto missing
-call "%~dp0bin\amd64\vcvarsamd64.bat"
+REM "%~dp0bin\amd64\vcvarsamd64.bat"
+if not exist "%~dp0bin\vcvars64.bat" goto missing
+call "%~dp0bin\vcvars64.bat"
 goto :eof

 :ia64
```
      1. GOTO _Verify that vcvarsall.bat initializes correctly for x64_ step
  1. You should be able to do a build now.