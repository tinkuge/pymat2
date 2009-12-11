import os
import re
import warnings

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

    def start(self):
        """Start Matlab instance.

        Return 'True' on success.

        """
        if not self.running:
            self._pymatMatlab.start()
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

    def plainEval(self, cmd):
        """Do a plain eval of given string in matlab.

        Does not perform fancy result parsing, etc.

        """
        if not self.running:
            raise exceptions.MatlabBackendStateError(
                "Matlab process not started")
        return self._pymatMatlab.eval(cmd)

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

    _returnValueRe = re.compile(r"^\s*(\w+)\s*=\s*((?:[0-9.eE+-]+\s*)+)", re.M)
    def eval(self, msg):
        """Eval given string and parse result.

        This eval() only parses and extracts variable names
        and values (in string form).

        """
        _out = {}
        _rv = self.plainEval(msg)
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
            for (_name, _value) in self.eval(cmd).iteritems()
        )

    def getArray(self, name):
        """Get given array from Matlab.

        Note that array is returned as numpy array.

        """
        if not self.running:
            raise exceptions.MatlabBackendStateError(
                "Matlab process not started")
        return self._pymatMatlab.getArray(name)

    def putArray(self, name, value):
        """Put given array from Matlab.

        Note that array is returned as numpy array.

        """
        if not self.running:
            raise exceptions.MatlabBackendStateError(
                "Matlab process not started")
        return self._pymatMatlab.putArray(name, value)

