In case of any problems, please refer to the [WindowsTips](WindowsTips.md) and [OsXtips](.md) page for hints on how to fix stuff on Windows and OS X respectivly.

# Tutorial #

After pymat2 module installation you should be able to import it as any other
Python module
```
Python 2.6.4 (r264:75708, Oct 26 2009, 08:23:19) [MSC v.1500 32 bit (Intel)] on
win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import pymat2
>>>
```

This module contains one most important object in whole pymat2 module - the "Matlab" object.
This object is meant to represent Matlab program instance in your Python session.
Given objects' constructor takes one optional argument - the string that determines how
Matlab program will be run on Unix-like OSes. Note that given argument is not required when running Matlab on Windows operating systems and issuing it on these systems will raise a Warning statement:
```
>>> _matlab = pymat2.Matlab("This is an argument")
C:\Python26\lib\site-packages\pymat2\Matlab.py:36: MatlabWarning: Init argument
is ignored on Windows.
  exceptions.MatlabWarning
```

On Windows Matlab is instantiated using ActiveX environment and if program fails to do this You should run
```
matlab.exe /regserver
```

From Windows command prompt. (Of course you must firstly must find appropriate Matlab executable on your system).

## Connecting to Matlab ##

Okay. So, after you create Matlab object instance in your Python interpreter,
```
>>> _matlab = pymat2.Matlab()
```
you should start connection to Matlab process.
This is done via issuing
```
>>> _matlab.start()
True
```
Note that there are `stop()`, `restart()` Matlab process-related functions and `.running`
with expected meanings.

## Interacting with Matlab ##

### .eval() ###
There is `Matlab.eval(<str>)` function that does exactly what you (or at least I) would expect it to do - evaluates given string in Matlab and returns Matlab output in string format.

For example, issuing `eval()` in Python:
```
>>> print _matlab.eval("1+1")

ans =

     2


>>>
```
and actually executing given command in Matlab command window:
```
  To get started, select MATLAB Help or Demos from the Help menu.

>> 1+1

ans =

     2

>> 
```

Please note that `eval()` is pretty dumb function - it won't check for syntax errors or raise any exceptions:
```
>>> print _matlab.eval("1+!")
??? Error: Unexpected MATLAB operator.


>>>
```

### .evalAndParse() ###

`Matlab.evalAndParse()` is slightly more sophisticated function - it actually tries to parse Matlab output (and it even sometimes succeeds doing this).

Given function returns Python dictionary where keys are variable names and values are unparsed variable values ("unparsed" means "in string form" in current context).
For example:
```
>>> _matlab.evalAndParse("1+1")
{'ans': '2'}
```
or more sophisticated:
```
>>> _matlab.evalAndParse("x=1+1;y=x+1;z=y*42;x,y,z")
{'y': '3', 'x': '2', 'z': '126'}
```
Given function tries to make sense from any kind of Matlab output and to act appropriately:
```
>>> _matlab.evalAndParse("1+!")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Python26\lib\site-packages\pymat2\Matlab.py", line 100, in evalAndPar
se
    (msg, _rv)
pymat2.exceptions.MatlabEvalError: Eval('1+!') error. Matlab answer:
??? Error: Unexpected MATLAB operator.

```

### .mapEval() ###

`Matlab.mapEval(<msg>, func)` is same as `Matlab.evalAndParse(<msg>)` with successive call
to `func(<value>)` for each value of output dictionary of `Matlab.mapEval()`.

### .putArray() ###

`Matlab.putArray(<name>, <value>)` puts NumPy array `<value>` to the Matlab variable `<name>`.

Please note that only 1-D and 2-D arrays are currently supported.

### .getArray() ###

`Matlab.getArray(<name>)` fetches and returns Matlab array with `<name>` as NumPy array.

## Misc functionality ##

### .version ###

`Matlab.version` constant contains **PyMat2** version, not Matlab version.

### .outputBufferSize ###

The actual C backend uses fixed-length string for obtaining Matlab output (one that is returned via `eval()` call, for example), and `Matlab.outputBufferSize` propery
provides way of reading current length of given array and setting it (setting this variable changes actual length of this buffer).

It your responsibility to ensure that length of expected Matlab output will be always smaller than length of given buffer. If you fail to do that something bad will happen.