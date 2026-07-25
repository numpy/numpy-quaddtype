import numpy as np
import pytest

from numpy_quaddtype import QuadPrecDType, QuadPrecision
from numpy_quaddtype._quaddtype_main import from_raw_bytes


def x87_longdouble_dtype():
    dtype = QuadPrecDType(backend="longdouble")
    if dtype.itemsize != 16 or np.finfo(np.longdouble).nmant != 63:
        pytest.skip("long double is not x87 80-bit stored in 16 bytes")
    return dtype


def poisoned_array(dtype, offset=0):
    storage = bytearray(b"\xa5" * (offset + dtype.itemsize))
    array = np.ndarray((1,), dtype=dtype, buffer=storage, offset=offset)
    return array, storage


def assert_zero_padding(storage, offset=0):
    assert bytes(storage[offset + 10 : offset + 16]) == b"\x00" * 6


def assert_array_padding_zero(array):
    raw = array.tobytes()
    for start in range(0, len(raw), array.dtype.itemsize):
        assert raw[start + 10 : start + 16] == b"\x00" * 6


@pytest.mark.parametrize("offset", [0, 1])
def test_setitem_zeroes_longdouble_padding(offset):
    dtype = x87_longdouble_dtype()
    array, storage = poisoned_array(dtype, offset)

    array[0] = QuadPrecision("1.5", backend="longdouble")

    assert array[0] == QuadPrecision("1.5", backend="longdouble")
    assert_zero_padding(storage, offset)


@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda value, out: np.negative(value, out=out), -1.5),
        (lambda value, out: np.add(value, value, out=out), 3.0),
        (lambda value, out: np.ldexp(value, 2, out=out), 6.0),
    ],
)
def test_ufunc_zeroes_longdouble_padding(offset, operation, expected):
    dtype = x87_longdouble_dtype()
    value = np.array([1.5], dtype=dtype)
    out, storage = poisoned_array(dtype, offset)

    operation(value, out)

    assert out[0] == QuadPrecision(str(expected), backend="longdouble")
    assert_zero_padding(storage, offset)


@pytest.mark.parametrize("offset", [0, 1])
def test_multi_output_ufunc_zeroes_longdouble_padding(offset):
    dtype = x87_longdouble_dtype()
    value = np.array([1.5], dtype=dtype)
    fractional, fractional_storage = poisoned_array(dtype, offset)
    integral, integral_storage = poisoned_array(dtype, offset)

    np.modf(value, out=(fractional, integral))

    assert fractional[0] == QuadPrecision("0.5", backend="longdouble")
    assert integral[0] == QuadPrecision("1.0", backend="longdouble")
    assert_zero_padding(fractional_storage, offset)
    assert_zero_padding(integral_storage, offset)


@pytest.mark.parametrize("offset", [0, 1])
def test_binary_multi_output_ufunc_zeroes_longdouble_padding(offset):
    dtype = x87_longdouble_dtype()
    dividend = np.array([5.5], dtype=dtype)
    divisor = np.array([2.0], dtype=dtype)
    quotient, quotient_storage = poisoned_array(dtype, offset)
    remainder, remainder_storage = poisoned_array(dtype, offset)

    np.divmod(dividend, divisor, out=(quotient, remainder))

    assert quotient[0] == QuadPrecision("2.0", backend="longdouble")
    assert remainder[0] == QuadPrecision("1.5", backend="longdouble")
    assert_zero_padding(quotient_storage, offset)
    assert_zero_padding(remainder_storage, offset)


@pytest.mark.parametrize("offset", [0, 1])
def test_frexp_zeroes_longdouble_padding(offset):
    dtype = x87_longdouble_dtype()
    value = np.array([6.0], dtype=dtype)
    mantissa, storage = poisoned_array(dtype, offset)
    exponent = np.empty(1, dtype=np.int32)

    np.frexp(value, out=(mantissa, exponent))

    assert mantissa[0] == QuadPrecision("0.75", backend="longdouble")
    assert exponent[0] == 3
    assert_zero_padding(storage, offset)


@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("source_backend", [None, "sleef"])
def test_cast_zeroes_longdouble_padding(offset, source_backend):
    dtype = x87_longdouble_dtype()
    if source_backend is None:
        source = np.array([1.5], dtype=np.float64)
    else:
        source = np.array(
            [QuadPrecision("1.5", backend=source_backend)],
            dtype=QuadPrecDType(backend=source_backend),
        )
    out, storage = poisoned_array(dtype, offset)

    np.copyto(out, source, casting="unsafe")

    assert out[0] == QuadPrecision("1.5", backend="longdouble")
    assert_zero_padding(storage, offset)


def test_arange_zeroes_longdouble_padding():
    dtype = x87_longdouble_dtype()

    result = np.arange(8, dtype=dtype)

    np.testing.assert_array_equal(
        result.astype(np.float64), np.arange(8, dtype=np.float64)
    )
    assert_array_padding_zero(result)


def test_fromstring_zeroes_longdouble_padding():
    dtype = x87_longdouble_dtype()

    result = np.fromstring("1.5 inf -nan", dtype=dtype, sep=" ")

    assert result[0] == QuadPrecision("1.5", backend="longdouble")
    assert np.isposinf(result[1])
    assert np.isnan(result[2])
    assert np.signbit(result[2])
    assert_array_padding_zero(result)


def test_from_raw_bytes_zeroes_longdouble_padding():
    x87_longdouble_dtype()
    original = QuadPrecision("1.5", backend="longdouble")
    raw = bytearray(original.__reduce__()[1][0])
    raw[10:] = b"\xa5" * 6

    result = from_raw_bytes(bytes(raw), "longdouble")
    result_raw = result.__reduce__()[1][0]

    assert result == original
    assert result_raw[10:] == b"\x00" * 6
