import os
import re
import warnings

import time

from . import (
    _pymat2,
    exceptions,
)

class Matlab(object):
    """Pretty python object to be used as Matlab interface."""

    _pymatMatlab = None

    version = property(
        lambda s: _pymat2.__version__)
    running = property(
        lambda s: s._pymatMatlab and s._pymatMatlab.isRunning())
    outputBufferSize = property(
        # getter
        lambda s: s._pymatMatlab and s._pymatMatlab.outputBufferSize,
        # setter
        lambda s, v: s._pymatMatlab and \
            s._pymatMatlab.setOutputBufferSize(v)
    )

    def __init__(self, arg=None):
        """Create Matlab interface instance.

        Argument to be passed only on Unixes.
        
        """
        if arg and os.name == "nt":
            warnings.warn(
                "Init argument is ignored on Windows.",
                 exceptions.MatlabWarning
            )
            arg = None
        self._pymatMatlab = _pymat2.Matlab(arg)

    def start(self, retryCount=20):
        """Start Matlab instance.

        Return 'True' on success.

        """
        _startupErrorCode = _pymat2.PymatErrorCodes["PYMAT_ERR_MATLAB_ENGINE_START"]
        if not self.running:
            for _attempt in xrange(retryCount):
                # This is a dirty hack, someone should
                # propose something better
                try:
                    self._pymatMatlab.start()
                except _pymat2.PymatError, _err:
                    if _err[0] == _startupErrorCode:
                        # Engine start failure - suppress it
                        # and probably attempt to start it after
                        # some delay
                        time.sleep(1)
                        pass
                    else:
                        # Unexpected error
                        raise
                else:
                    break
            else:
                # "for" loop didn't end with "break" call
                # engine startup failed for sure.
                raise exceptions.MatlabStartupError("All attempts to start engoine failed.")
        return self.running

    def stop(self):
        """Stop Matlab.

        Return 'True' on success.
        """
        if self.running:
            self._pymatMatlab.stop()
        return not self.running

    def restart(self):
        """Restart Matlab."""
        return self.stop() and self.start()

    def _eval(self, cmd):
        """Do a plain eval of given string in matlab.

        Does not perform fancy result parsing, etc.

        """
        if not self.running:
            raise exceptions.MatlabBackendStateError(
                "Matlab process not started")
        return self._pymatMatlab.eval(cmd)

    def eval(self, cmd):
        _rv = self._eval(cmd)
        if _rv.startswith(">>"):
            # matlab return on *nix starts with theese '>>'
            # removing them to make output truly cross-platrofm
            _rv = _rv.lstrip('>')
        return _rv.lstrip()

    def _parseMatlabOutput(self, value, mapType):
        # Please note that parsing is quite simple.
        # It won't handle full range of Matlab outputs
        _lines = [_el.split() for _el in value.splitlines() if _el.strip()]
        print _lines
        if len(_lines) == 1 and len(_lines[0]) == 1:
            # Assume that is only one value returned,
            # Than it must be single value not
            # and array with single element
            _out = mapType(_lines[0][0])
        return _out

    _returnValueRe = re.compile(r"^(\w+)\s*=\s*((?:[0-9.eE+-]+\s*)+)", re.M)
    def evalAndParse(self, msg):
        """Eval given string and parse result.

        This eval() only parses and extracts variable names
        and values (in string form).

        """
        _out = {}
        _rv = self.eval(msg)
        if _rv.startswith("???"):
            raise exceptions.MatlabEvalError(
                "Eval(%r) error. Matlab answer:\n%s" % 
                    (msg, _rv)
            )
        while _rv:
            _match = self._returnValueRe.match(_rv)
            if not _match:
                raise exceptions.ResiultParsingError(
                    "Failed to parse result string %r" % _rv)
            (_name, _value) = _match.groups()
            _out[_name] = _value.strip()
            _rv = _rv[_match.end():].lstrip()
        return _out

    def mapEval(self, cmd, parser):
        """Eval() given command in Matlab and apply given parser()
        callable to each element of result.

        """
        return dict((_name, parser(_value))
            for (_name, _value) in self.evalAndParse(cmd).iteritems()
        )

    def getArray(self, name):
        """Get given array from Matlab.

        Note that array is returned as numpy array.

        """
        if not self.running:
            raise exceptions.MatlabBackendStateError(
                "Matlab process not started")
        _rv = self._pymatMatlab.getArray(name)
        if len(_rv) == 1:
            # Array is really 1-D
            _rv = _rv[0]
        return _rv

    def putArray(self, name, value):
        """Put given array from Matlab.

        Note that array is returned as numpy array.

        """
        if not self.running:
            raise exceptions.MatlabBackendStateError(
                "Matlab process not started")
        return self._pymatMatlab.putArray(name, value)

    def __del__(self):
        """Gracefully close Matlab session on object deletion."""
        _matlabObject = self._pymatMatlab
        if _matlabObject and _matlabObject.isRunning():
            _matlabObject.stop()
