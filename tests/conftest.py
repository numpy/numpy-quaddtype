# Pre-import numpy's lazy submodules in single-threaded context before
# pytest-run-parallel forks workers. Works around CPython issue https://github.com/python/cpython/issues/149728 
# which causes numpy's lazy __getattr__ on rec/ma/linalg/fft/... to recurse
# to RecursionError under concurrent first-touch from multiple threads.
# Drop this file when CPython bugfix release happens.

import numpy

for _attr in (
    "linalg", "fft", "dtypes", "random", "polynomial", "ma",
    "ctypeslib", "exceptions", "testing", "matlib", "f2py",
    "typing", "rec", "char", "core", "strings",
):
    getattr(numpy, _attr)
del _attr
