from numpy import *

def doLog(msg):
    print(msg)

def check(msg, testCall):
    doLog("- %s" % msg)
    if not testCall():
        raise Exception("Test Failed")
    doLog("\t-> Success.")


def doTest(module):
    doLog("Starting test of %s..." % module)

    doLog("Creating Matlab object instance. Raises a Warning on Windows.")
    _obj = module.Matlab("Something")
    _desiredVersion = file("pymat2.version").read().strip()
    check("Pymat current version (%s) is %s" % (_obj.version, _desiredVersion),
        lambda: _obj.version == _desiredVersion
    )
    check(
        "Matlab must be not running now.",
        lambda: _obj.running == False
    )
    check(
        "Starting matlab.",
        lambda: _obj.start() == True
    )
    check(
        "Checking that output buffer size can be read.",
        lambda: _obj.outputBufferSize is not None
    )
    def _bufferCheck():
        _size = _obj.outputBufferSize
        _obj.outputBufferSize = _size + 42
        return _obj.outputBufferSize == _size + 42

    check(
        "Checking that output buffer size can be written.",
        _bufferCheck
    )
    check(
        "Checking plain eval.",
        lambda: _obj.eval("x=1+1; y=x+1; y") == "\ny =\n\n     3\n\n"
    )
    check(
        "Checking fancy eval with default mapping.",
        lambda: _obj.evalAndParse("x=1+1; y=x+1; y") == {"y": "3"}
    )
    check(
        "Checking fancy eval with default mapping. With 3 vars.",
        lambda: _obj.evalAndParse("x=1+1; y=x+1; z=x+y; x, y, z") \
            == {"x": "2", "y": "3", "z": "5"}
    )
    check(
        "Checking eval float mapping.",
        lambda: abs(_obj.mapEval("3/2", float)["ans"] - 1.5) < 0.0001
    )
    def _putArrayCheck():
        _myArr = range(42)
        _arr = array(_myArr)
        _obj.putArray("arr1", _arr)
        return _obj.mapEval(
            "sum(arr1)", 
            lambda val: map(int, val.split())
        )["ans"] == sum(_myArr)
    check(
        "Checking put array.",
        _putArrayCheck
    )
    def _getArrayCheck():
        _obj.putArray("arr2", array(xrange(10)))
        _obj.eval("outArr = dct(arr2)")
        return abs(sum(_obj.getArray("outArr")) - 3.75878598276) < 0.00001
    check(
        "Checking get array.",
        _getArrayCheck
    )
    check(
        "Checking .stop()",
        lambda: _obj.stop() == True # I want to be obvious here.
    )
    doLog("Test ended.")

if __name__ == "__main__":
    import src as pymat
    doTest(pymat)
