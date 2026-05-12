import numpy

for _attr in (
    "linalg", "fft", "dtypes", "random", "polynomial", "ma",
    "ctypeslib", "exceptions", "testing", "matlib", "f2py",
    "typing", "rec", "char", "core", "strings",
):
    getattr(numpy, _attr)
del _attr
