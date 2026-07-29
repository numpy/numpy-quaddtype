import ctypes

import numpy as np
import pytest

from numpy_quaddtype import QuadPrecDType, QuadPrecision
from numpy_quaddtype._quaddtype_main import from_raw_bytes


X87_LONG_DOUBLE_VALUE_BYTES = 10
X87_LONG_DOUBLE_STORAGE_BYTES = 16


def x87_longdouble_dtype():
    dtype = QuadPrecDType(backend="longdouble")
    if (
        dtype.itemsize != X87_LONG_DOUBLE_STORAGE_BYTES
        or np.finfo(np.longdouble).nmant != 63
    ):
        pytest.skip("long double is not x87 80-bit stored in 16 bytes")
    return dtype


def poisoned_array(dtype, aligned):
    storage = bytearray(b"\xa5" * (dtype.itemsize + dtype.alignment))
    address = ctypes.addressof(ctypes.c_char.from_buffer(storage))
    offset = (-address) % dtype.alignment
    if not aligned:
        offset += 1
    array = np.ndarray((1,), dtype=dtype, buffer=storage, offset=offset)
    assert array.flags.aligned == aligned
    return array, storage, offset


def assert_zero_padding(storage, offset=0):
    padding = storage[
        offset + X87_LONG_DOUBLE_VALUE_BYTES : offset + X87_LONG_DOUBLE_STORAGE_BYTES
    ]
    assert bytes(padding) == b"\x00" * (
        X87_LONG_DOUBLE_STORAGE_BYTES - X87_LONG_DOUBLE_VALUE_BYTES
    )


def assert_array_padding_zero(array):
    raw = array.tobytes()
    for start in range(0, len(raw), array.dtype.itemsize):
        assert_zero_padding(raw, start)


def test_scalar_construction_zeroes_longdouble_padding():
    x87_longdouble_dtype()

    value = QuadPrecision("1.5", backend="longdouble")

    assert_zero_padding(value.__reduce__()[1][0])


@pytest.mark.parametrize("aligned", [True, False])
def test_setitem_zeroes_longdouble_padding(aligned):
    dtype = x87_longdouble_dtype()
    array, storage, offset = poisoned_array(dtype, aligned)

    array[0] = QuadPrecision("1.5", backend="longdouble")

    assert array[0] == QuadPrecision("1.5", backend="longdouble")
    assert_zero_padding(storage, offset)


@pytest.mark.parametrize("aligned", [True, False])
@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda value, out: np.negative(value, out=out), -1.5),
        (lambda value, out: np.add(value, value, out=out), 3.0),
        (lambda value, out: np.ldexp(value, 2, out=out), 6.0),
    ],
)
def test_ufunc_zeroes_longdouble_padding(aligned, operation, expected):
    dtype = x87_longdouble_dtype()
    value = np.array([1.5], dtype=dtype)
    out, storage, offset = poisoned_array(dtype, aligned)

    operation(value, out)

    assert out[0] == QuadPrecision(str(expected), backend="longdouble")
    assert_zero_padding(storage, offset)


@pytest.mark.parametrize("aligned", [True, False])
def test_multi_output_ufunc_zeroes_longdouble_padding(aligned):
    dtype = x87_longdouble_dtype()
    value = np.array([1.5], dtype=dtype)
    fractional, fractional_storage, fractional_offset = poisoned_array(dtype, aligned)
    integral, integral_storage, integral_offset = poisoned_array(dtype, aligned)

    np.modf(value, out=(fractional, integral))

    assert fractional[0] == QuadPrecision("0.5", backend="longdouble")
    assert integral[0] == QuadPrecision("1.0", backend="longdouble")
    assert_zero_padding(fractional_storage, fractional_offset)
    assert_zero_padding(integral_storage, integral_offset)


@pytest.mark.parametrize("aligned", [True, False])
def test_binary_multi_output_ufunc_zeroes_longdouble_padding(aligned):
    dtype = x87_longdouble_dtype()
    dividend = np.array([5.5], dtype=dtype)
    divisor = np.array([2.0], dtype=dtype)
    quotient, quotient_storage, quotient_offset = poisoned_array(dtype, aligned)
    remainder, remainder_storage, remainder_offset = poisoned_array(dtype, aligned)

    np.divmod(dividend, divisor, out=(quotient, remainder))

    assert quotient[0] == QuadPrecision("2.0", backend="longdouble")
    assert remainder[0] == QuadPrecision("1.5", backend="longdouble")
    assert_zero_padding(quotient_storage, quotient_offset)
    assert_zero_padding(remainder_storage, remainder_offset)


@pytest.mark.parametrize("aligned", [True, False])
def test_frexp_zeroes_longdouble_padding(aligned):
    dtype = x87_longdouble_dtype()
    value = np.array([6.0], dtype=dtype)
    mantissa, storage, offset = poisoned_array(dtype, aligned)
    exponent = np.empty(1, dtype=np.int32)

    np.frexp(value, out=(mantissa, exponent))

    assert mantissa[0] == QuadPrecision("0.75", backend="longdouble")
    assert exponent[0] == 3
    assert_zero_padding(storage, offset)


@pytest.mark.parametrize("aligned", [True, False])
@pytest.mark.parametrize("source_backend", [None, "sleef"])
def test_cast_zeroes_longdouble_padding(aligned, source_backend):
    dtype = x87_longdouble_dtype()
    if source_backend is None:
        source = np.array([1.5], dtype=np.float64)
    else:
        source = np.array(
            [QuadPrecision("1.5", backend=source_backend)],
            dtype=QuadPrecDType(backend=source_backend),
        )
    out, storage, offset = poisoned_array(dtype, aligned)

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
    raw[X87_LONG_DOUBLE_VALUE_BYTES:] = b"\xa5" * (
        X87_LONG_DOUBLE_STORAGE_BYTES - X87_LONG_DOUBLE_VALUE_BYTES
    )

    result = from_raw_bytes(bytes(raw), "longdouble")
    result_raw = result.__reduce__()[1][0]

    assert result == original
    assert_zero_padding(result_raw)
