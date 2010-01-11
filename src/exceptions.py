class MatlabWarning(Warning):
    """Base Warning class."""

class MatlabError(Exception):
    """Base calss for all Pymat matlab exceptions."""

class MatlabBackendStateError(MatlabError):
    """Risen when something wrong is happended with matlab process itself."""

class ResiultParsingError(MatlabError):
    """Risen when matlab result parsing failed."""

class MatlabEvalError(MatlabError):
    """Risen when matlab eval() returns an error."""

class MatlabStartupError(MatlabError):
    """Risen when matlab engine failes to start."""
