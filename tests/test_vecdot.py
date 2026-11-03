from fractions import Fraction

import numpy as np
import pytest
from numpy.testing._private.utils import run_threaded

from numpy_quaddtype import QuadPrecDType, QuadPrecision

BACKENDS = ["sleef", "longdouble"]
UNIT_ROUNDOFF = Fraction(1, 2**113)


def _quad(values, backend):
    return np.asarray(values).astype(QuadPrecDType(backend=backend))


def _random(rng, shape):
    return rng.standard_normal(shape) * 2.0 ** rng.integers(-30, 31, shape)


def _to_fraction(value):
    return Fraction(*value.as_integer_ratio())


def _assert_vecdot(result, x, y, backend, axis=-1):
    """Check ``result`` against ``np.vecdot(x, y, axis=axis)`` of float64 inputs.

    The longdouble backend sums in the same order as NumPy's longdouble loop,
    so it must match bit for bit. SLEEF results must lie within the standard
    dot-product error bound of the exact rational result.
    """
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    y = np.moveaxis(np.asarray(y, dtype=np.float64), axis, -1)
    x, y = np.broadcast_arrays(x, y)
    result = np.asarray(result)
    assert result.dtype == QuadPrecDType(backend=backend)
    assert result.shape == x.shape[:-1]

    if backend == "longdouble":
        expected = np.vecdot(x.astype(np.longdouble), y.astype(np.longdouble))
        np.testing.assert_array_equal(result.astype(np.longdouble), expected,
                                      strict=True)
        return

    n = x.shape[-1]
    for idx in np.ndindex(result.shape):
        terms = [Fraction(a) * Fraction(b) for a, b in zip(x[idx], y[idx])]
        bound = 2 * n * UNIT_ROUNDOFF * sum(abs(t) for t in terms)
        error = abs(_to_fraction(result[idx]) - sum(terms, Fraction(0)))
        assert error <= bound, f"{idx}: error {float(error)!r} > {float(bound)!r}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_vector_inputs_give_scalar(backend):
    result = np.vecdot(_quad([1, 2, 3], backend), _quad([4, 5, 6], backend))
    assert isinstance(result, QuadPrecision)
    assert result == QuadPrecision(32, backend=backend)
    _assert_vecdot(result, [1, 2, 3], [4, 5, 6], backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n", [1, 2, 7, 64, 1000])
def test_contiguous_matches_reference(backend, n):
    rng = np.random.default_rng(n)
    x, y = _random(rng, n), _random(rng, n)
    _assert_vecdot(np.vecdot(_quad(x, backend), _quad(y, backend)), x, y, backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("layout", ["step2", "x_reversed", "y_reversed", "x_broadcast",
                                    "y_broadcast", "transposed", "fortran"])
def test_core_layouts(backend, layout):
    rng = np.random.default_rng(3)
    x, y = _random(rng, (4, 9)), _random(rng, (4, 9))
    qx, qy = _quad(x, backend), _quad(y, backend)
    axis = -1
    if layout == "step2":
        qx, qy, x, y = qx[:, ::2], qy[:, ::2], x[:, ::2], y[:, ::2]
    elif layout == "x_reversed":
        qx, x = qx[:, ::-1], x[:, ::-1]
    elif layout == "y_reversed":
        qy, y = qy[:, ::-1], y[:, ::-1]
    elif layout == "x_broadcast":
        qx, x = np.broadcast_to(qx[:, :1], qx.shape), np.broadcast_to(x[:, :1], x.shape)
    elif layout == "y_broadcast":
        qy, y = np.broadcast_to(qy[:, :1], qy.shape), np.broadcast_to(y[:, :1], y.shape)
    elif layout == "transposed":
        axis = 0
    elif layout == "fortran":
        qx, qy = np.asfortranarray(qx), np.asfortranarray(qy)
    _assert_vecdot(np.vecdot(qx, qy, axis=axis), x, y, backend, axis=axis)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("x_shape,y_shape", [
    ((3, 5), (3, 5)),
    ((4, 5), (5,)),
    ((2, 1, 5), (1, 3, 5)),
    ((2, 3, 4, 5), (4, 5)),
])
def test_outer_dimensions_broadcast(backend, x_shape, y_shape):
    rng = np.random.default_rng(4)
    x, y = _random(rng, x_shape), _random(rng, y_shape)
    _assert_vecdot(np.vecdot(_quad(x, backend), _quad(y, backend)), x, y, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_axes_and_keepdims(backend):
    x = np.arange(6.0).reshape(2, 3)
    y = np.arange(6.0).reshape(3, 2)
    result = np.vecdot(_quad(x, backend), _quad(y, backend), axes=[(1,), (0,), ()])
    _assert_vecdot(result, x, y.T, backend)

    kept = np.vecdot(_quad(x, backend), _quad([1, 2, 3], backend), keepdims=True)
    assert kept.shape == (2, 1)
    _assert_vecdot(kept[:, 0], x, [1, 2, 3], backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("out_layout", ["contiguous", "strided"])
def test_out(backend, out_layout):
    rng = np.random.default_rng(5)
    x, y = _random(rng, (6, 8)), _random(rng, (6, 8))
    dtype = QuadPrecDType(backend=backend)
    out = np.full(12 if out_layout == "strided" else 6, 999, dtype=dtype)
    if out_layout == "strided":
        out = out[::2]

    result = np.vecdot(_quad(x, backend), _quad(y, backend), out=out)
    assert result is out
    _assert_vecdot(out, x, y, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_unaligned_operands(backend):
    dtype = QuadPrecDType(backend=backend)

    def unaligned(shape):
        size = int(np.prod(shape))
        return np.ndarray(shape, dtype=dtype, offset=1,
                          buffer=bytearray(1 + size * dtype.itemsize))

    rng = np.random.default_rng(6)
    x, y = _random(rng, (3, 7)), _random(rng, (3, 7))
    qx, qy, out = unaligned((3, 7)), unaligned((3, 7)), unaligned((3,))
    qx[...], qy[...] = _quad(x, backend), _quad(y, backend)
    assert not (qx.flags.aligned or qy.flags.aligned or out.flags.aligned)

    assert np.vecdot(qx, qy, out=out) is out
    _assert_vecdot(out, x, y, backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", [(0,), (3, 0), (2, 1, 0)])
def test_empty_core_dimension_is_zero(backend, shape):
    dtype = QuadPrecDType(backend=backend)
    out = np.full(shape[:-1], 999, dtype=dtype)
    np.vecdot(np.empty(shape, dtype), np.empty(shape, dtype), out=out)
    np.testing.assert_array_equal(out, np.zeros(shape[:-1], dtype=dtype))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("x,y,expected", [
    ([np.inf, 1], [1, 2], np.inf),
    ([-np.inf, 1], [1, 2], -np.inf),
    ([np.inf, 1], [0, 2], np.nan),
    ([np.inf, -np.inf], [1, 1], np.nan),
    ([np.nan, 1], [1, 2], np.nan),
])
def test_special_values(backend, x, y, expected):
    with np.errstate(invalid="ignore"):
        result = np.vecdot(_quad(x, backend), _quad(y, backend))
    np.testing.assert_array_equal(np.float64(result), expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cancellation_beyond_float64(backend):
    x, y = [2.0**100, 1.0, -(2.0**100)], [1.0, 1.0, 1.0]
    result = np.vecdot(_quad(x, backend), _quad(y, backend))
    if backend == "sleef":
        assert result == QuadPrecision(1)
    _assert_vecdot(result, x, y, backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("layout", ["contiguous", "step2", "reversed"])
def test_matches_matmul_for_vectors(backend, layout):
    rng = np.random.default_rng(7)
    qx, qy = _quad(_random(rng, 40), backend), _quad(_random(rng, 40), backend)
    if layout == "step2":
        qx, qy = qx[::2], qy[::2]
    elif layout == "reversed":
        qx, qy = qx[::-1], qy[::-1]
    v, m = np.vecdot(qx, qy), np.matmul(qx, qy)
    assert v.as_integer_ratio() == m.as_integer_ratio()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("other", [
    np.array([4.0, 5.0, 6.0]),
    np.array([4, 5, 6]),
    [4, 5, 6],
])
@pytest.mark.parametrize("quad_first", [True, False])
def test_builtin_operand_promotes_to_quad(backend, other, quad_first):
    quad = _quad([1, 2, 3], "sleef")
    x, y = (quad, other) if quad_first else (other, quad)
    result = np.vecdot(x, y)
    assert result == QuadPrecision(32)
    assert result.dtype == QuadPrecDType(backend="sleef")


@pytest.mark.parametrize("quad_first", [True, False])
def test_object_operand_uses_object_loop(quad_first):
    quad = _quad([1, 2], "sleef")
    objects = np.array([3, 4], dtype=object)
    x, y = (quad, objects) if quad_first else (objects, quad)
    assert np.result_type(x, y) == np.dtype(object)
    assert np.vecdot(x, y) == QuadPrecision(11)
    assert np.vecdot(np.stack([x, x]), y).dtype == np.dtype(object)


@pytest.mark.parametrize("x_backend,y_backend,out_backend", [
    ("sleef", "longdouble", None),
    ("longdouble", "sleef", None),
    ("sleef", "sleef", "longdouble"),
    ("longdouble", "longdouble", "sleef"),
])
def test_mixed_backends_raise(x_backend, y_backend, out_backend):
    kwargs = {}
    if out_backend is not None:
        kwargs["out"] = np.empty((), dtype=QuadPrecDType(backend=out_backend))
    with pytest.raises(NotImplementedError, match="same backend"):
        np.vecdot(_quad([1, 2], x_backend), _quad([3, 4], y_backend), **kwargs)


@pytest.mark.parametrize("quad_first", [True, False])
def test_longdouble_with_builtin_operand_raises(quad_first):
    # Builtin operands become SLEEF descriptors, which is a mixed-backend call (gh-113).
    quad = _quad([1, 2], "longdouble")
    other = np.array([3.0, 4.0])
    x, y = (quad, other) if quad_first else (other, quad)
    with pytest.raises(NotImplementedError, match="same backend"):
        np.vecdot(x, y)


@pytest.mark.parametrize("backend", BACKENDS)
def test_shape_errors(backend):
    with pytest.raises(ValueError, match="core dimension"):
        np.vecdot(_quad([1, 2, 3], backend), _quad([1, 2], backend))
    with pytest.raises(ValueError, match="does not have enough dimensions"):
        np.vecdot(_quad(1, backend), _quad([1], backend))


def test_longdouble_output_padding_is_zeroed():
    dtype = QuadPrecDType(backend="longdouble")
    if np.finfo(np.longdouble).nmant != 63 or dtype.itemsize != 16:
        pytest.skip("requires x87 80-bit long double stored in 16 bytes")
    storage = bytearray(b"\xa5" * dtype.itemsize)
    out = np.ndarray((), dtype=dtype, buffer=storage)
    np.vecdot(_quad([1, 2, 3], "longdouble"), _quad([4, 5, 6], "longdouble"), out=out)
    assert out == QuadPrecision(32, backend="longdouble")
    assert bytes(storage[10:]) == bytes(6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_concurrent_calls(backend):
    rng = np.random.default_rng(8)
    x, y = _random(rng, (16, 512)), _random(rng, (16, 512))
    qx, qy = _quad(x, backend), _quad(y, backend)
    expected = np.vecdot(qx, qy)

    def worker():
        for _ in range(5):
            np.testing.assert_array_equal(np.vecdot(qx, qy), expected)

    run_threaded(worker, max_workers=8)
