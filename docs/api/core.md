# Core Types

The fundamental types provided by NumPy QuadDType.

## Quad Precision Value

```{eval-rst}
.. class:: numpy_quaddtype.QuadPrecision(value, backend="sleef")

   A quad-precision (128-bit) floating-point scalar.

   QuadPrecision is a NumPy scalar type that provides IEEE 754 binary128
   floating-point arithmetic. It can be used standalone or as elements
   of NumPy arrays.

   :param value: The value to convert to quad precision. It can be:

       - ``float`` or ``int``: Python numeric types
       - ``str``: String representation for maximum precision
       - ``bytes``: Raw 16-byte representation
       - ``numpy.floating`` or ``numpy.integer``: NumPy numeric types
       - ``QuadPrecision``: Another QuadPrecision value
   :type value: float, int, str, bytes, numpy scalar, or QuadPrecision

   :param backend: Computation backend to use. Either ``"sleef"`` (default) 
       or ``"longdouble"``.
   :type backend: str, optional

   **Examples**

   Create from different input types::

       >>> from numpy_quaddtype import QuadPrecision
       >>> QuadPrecision(3.14)
       QuadPrecision('3.14000000000000012434...')
       >>> QuadPrecision("3.14159265358979323846264338327950288")
       QuadPrecision('3.14159265358979323846264338327950288')
       >>> QuadPrecision(42)
       QuadPrecision('42.0')

   Arithmetic operations::

       >>> x = QuadPrecision("1.5")
       >>> y = QuadPrecision("2.5")
       >>> x + y
       QuadPrecision('4.0')
       >>> x * y
       QuadPrecision('3.75')

   .. attribute:: dtype
      :type: QuadPrecDType

      The NumPy dtype for this scalar.

   .. attribute:: real
      :type: QuadPrecision

      The real part (always self for QuadPrecision).

   .. attribute:: imag
      :type: QuadPrecision

      The imaginary part (always zero for QuadPrecision).
```

## Quad Precision DType

```{eval-rst}
.. class:: numpy_quaddtype.QuadPrecDType(backend="sleef")

   NumPy dtype for quad-precision floating-point arrays.

   QuadPrecDType is a custom NumPy dtype that enables the creation and
   manipulation of arrays containing quad-precision values.

   :param backend: Computation backend. Either ``"sleef"`` (default) or
       ``"longdouble"``.
   :type backend: str, optional

   **Examples**

   Create arrays with QuadPrecDType::

       >>> import numpy as np
       >>> from numpy_quaddtype import QuadPrecDType
       >>> arr = np.array([1, 2, 3], dtype=QuadPrecDType())
       >>> arr.dtype
       QuadPrecDType128
       >>> np.zeros(5, dtype=QuadPrecDType())
       array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=QuadPrecDType128)

   .. attribute:: backend
      :type: QuadBackend

      The computation backend (``SLEEF`` or ``LONGDOUBLE``).

   .. attribute:: itemsize
      :type: int

      The size of each element in bytes (always 16).

   .. attribute:: alignment
      :type: int

      The memory alignment in bytes (always 16).

   .. attribute:: name
      :type: str

      The string name of the dtype (``"QuadPrecDType128"``).
```

```{eval-rst}
.. class:: numpy_quaddtype.QuadBackend

   Enumeration of available computation backends.

   .. attribute:: SLEEF
      :value: 0

      SLEEF library backend (default). Provides true IEEE 754 binary128
      quad precision with SIMD optimization.

   .. attribute:: LONGDOUBLE
      :value: 1

      The platform's native long double backend. The precision varies by platform.

   **Example**

   ::

       >>> from numpy_quaddtype import QuadPrecDType, QuadBackend
       >>> dtype = QuadPrecDType()
       >>> dtype.backend == QuadBackend.SLEEF
       True
```

### Mixing backends

Values never move between the SLEEF and longdouble backends: any operation
that would combine or convert them raises `TypeError`: either
`Cannot mix QuadPrecision backends ...` or, where NumPy checks the casting rule
first, NumPy's own `Cannot cast ...` message. This covers ufuncs,
`out=` arrays, `astype`, assignment, `np.concatenate`, `np.result_type` and
`QuadPrecision(value, backend=...)`. NumPy's own dtypes, including
`np.longdouble`, work with both backends.

Python `int` and `float` operands use the SLEEF backend, so with longdouble
arrays they raise too. This includes NumPy functions that pass Python scalars
internally, such as `np.ones`, `np.full`, `np.ones_like`, `mean`, `var`,
`std`, `np.median`, `np.isclose`, `np.allclose` and `np.round`. Pass NumPy scalars or arrays, or
`QuadPrecision(..., backend="longdouble")`, instead, for example
`np.full(3, np.longdouble(1), dtype=dt)` or `np.ones(3).astype(dt)`:

```python
>>> import numpy as np
>>> from numpy_quaddtype import QuadPrecDType
>>> ld = np.array([1.5, 2.5], dtype=QuadPrecDType(backend="longdouble"))
>>> ld + np.float64(1)
array([2.5, 3.5], dtype=QuadPrecDType(backend='longdouble'))
>>> ld + 1
Traceback (most recent call last):
  ...
TypeError: Cannot mix QuadPrecision backends 'sleef' and 'longdouble' (Python int and float operands use 'sleef')
>>> ld.astype(QuadPrecDType())
Traceback (most recent call last):
  ...
TypeError: Cannot mix QuadPrecision backends 'sleef' and 'longdouble' (Python int and float operands use 'sleef')
```

## Convenience Functions

```{eval-rst}
.. function:: numpy_quaddtype.SleefQuadPrecision(value)

   Create a QuadPrecision scalar using the SLEEF backend.

   Equivalent to ``QuadPrecision(value, backend="sleef")``.

   :param value: Value to convert to quad precision.
   :return: Quad precision scalar using SLEEF backend.
   :rtype: QuadPrecision
```

```{eval-rst}
.. function:: numpy_quaddtype.LongDoubleQuadPrecision(value)

   Create a QuadPrecision scalar using the longdouble backend.

   Equivalent to ``QuadPrecision(value, backend="longdouble")``.

   :param value: Value to convert to quad precision.
   :return: Quad precision scalar using longdouble backend.
   :rtype: QuadPrecision
```

```{eval-rst}
.. function:: numpy_quaddtype.SleefQuadPrecDType()

   Create a QuadPrecDType using the SLEEF backend.

   Equivalent to ``QuadPrecDType(backend="sleef")``.

   :return: Dtype for SLEEF-backed quad precision arrays.
   :rtype: QuadPrecDType
```

```{eval-rst}
.. function:: numpy_quaddtype.LongDoubleQuadPrecDType()

   Create a QuadPrecDType using the longdouble backend.

   Equivalent to ``QuadPrecDType(backend="longdouble")``.

   :return: Dtype for longdouble-backed quad precision arrays.
   :rtype: QuadPrecDType
```
