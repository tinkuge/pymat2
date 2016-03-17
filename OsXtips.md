It has been reported (by Christian Prinoth) that it is possible to compile and run pymat2 for OS X.

The catch is that you need to set DYLD\_LIBRARY\_PATH before launching Python, e.g.
```
export DYLD_LIBRARY_PATH=/Applications/MATLAB_R2009b.app/bin/maci64/
```