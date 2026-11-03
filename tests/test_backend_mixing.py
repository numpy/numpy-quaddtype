"""Values never move between the SLEEF and longdouble backends (gh-112, gh-113).

NumPy's own dtypes, including ``np.longdouble``, are not backends and keep
working with both.
"""
import numpy as np
import pytest

from numpy_quaddtype import QuadPrecDType, QuadPrecision

MIXED = "Cannot mix QuadPrecision backends"
PAIRS = [("sleef", "longdouble"), ("longdouble", "sleef")]
BACKENDS = ["sleef", "longdouble"]


def _arr(values, backend):
    return np.array(values, dtype=QuadPrecDType(backend=backend))


def _raises():
    return pytest.raises(TypeError, match=MIXED)


def _cast_refused():
    # Under a casting rule NumPy checks can_cast first and raises its own error.
    return pytest.raises(TypeError, match=f"{MIXED}|Cannot cast array data")


@pytest.mark.parametrize("src,dst", PAIRS)
class TestArraysOfDifferentBackends:
    @pytest.mark.parametrize("op", [
        np.add, np.subtract, np.multiply, np.true_divide, np.power, np.maximum,
        np.equal, np.less, np.logical_and, np.divmod,
    ])
    def test_binary_ufunc(self, src, dst, op):
        with _raises():
            op(_arr([1, 2], src), _arr([3, 4], dst))

    @pytest.mark.parametrize("casting", ["same_kind", "unsafe"])
    def test_binary_out(self, src, dst, casting):
        out = _arr([7, 7], dst)
        with _raises():
            np.add(_arr([1, 2], src), _arr([3, 4], src), out=out, casting=casting)
        np.testing.assert_array_equal(out, _arr([7, 7], dst))

    def test_divmod_out(self, src, dst):
        with _raises():
            np.divmod(_arr([7], src), _arr([2], src), out=(_arr([0], dst), _arr([0], src)))

    def test_ldexp_out(self, src, dst):
        with _raises():
            np.ldexp(_arr([1.5], src), np.array([2]), out=_arr([0], dst))

    @pytest.mark.parametrize("casting", ["same_kind", "unsafe"])
    @pytest.mark.parametrize("op", [np.sqrt, np.negative, np.sin])
    def test_unary_out(self, src, dst, op, casting):
        out = _arr([7, 7], dst)
        with _raises():
            op(_arr([0.5, 2.0], src), out=out, casting=casting)
        np.testing.assert_array_equal(out, _arr([7, 7], dst))

    @pytest.mark.parametrize("casting", ["same_kind", "unsafe"])
    def test_modf_out(self, src, dst, casting):
        frac, whole = _arr([7], dst), _arr([7], src)
        with _raises():
            np.modf(_arr([1.5], src), out=(frac, whole), casting=casting)
        np.testing.assert_array_equal(frac, _arr([7], dst))

    @pytest.mark.parametrize("casting", ["same_kind", "unsafe"])
    def test_frexp_out(self, src, dst, casting):
        mantissa = _arr([7], dst)
        with _raises():
            np.frexp(_arr([6.0], src), out=(mantissa, np.empty(1, np.int32)),
                     casting=casting)
        np.testing.assert_array_equal(mantissa, _arr([7], dst))

    def test_astype(self, src, dst):
        with _raises():
            _arr([1.5], src).astype(QuadPrecDType(backend=dst))

    @pytest.mark.parametrize("casting", ["same_kind", "safe", "same_value"])
    def test_astype_with_casting_rule(self, src, dst, casting):
        with _cast_refused():
            _arr([1.5], src).astype(QuadPrecDType(backend=dst), casting=casting)

    @pytest.mark.parametrize("convert", [
        lambda a, dt: np.asarray(a, dtype=dt),
        lambda a, dt: np.array(a, dtype=dt),
    ], ids=["asarray", "array"])
    def test_array_construction(self, src, dst, convert):
        with _raises():
            convert(_arr([1.5], src), QuadPrecDType(backend=dst))

    def test_slice_assignment(self, src, dst):
        target = _arr([7, 7], dst)
        with _raises():
            target[...] = _arr([1, 2], src)
        np.testing.assert_array_equal(target, _arr([7, 7], dst))

    def test_copyto(self, src, dst):
        target = _arr([7], dst)
        with _raises():
            np.copyto(target, _arr([1], src), casting="unsafe")
        np.testing.assert_array_equal(target, _arr([7], dst))

    @pytest.mark.parametrize("join", [np.concatenate, np.stack])
    def test_join(self, src, dst, join):
        with _raises():
            join([_arr([1], src), _arr([2], dst)])

    def test_promotion(self, src, dst):
        s, d = QuadPrecDType(backend=src), QuadPrecDType(backend=dst)
        with _raises():
            np.result_type(s, d)
        with _raises():
            np.promote_types(s, d)

    @pytest.mark.parametrize("casting", ["safe", "same_kind"])
    def test_can_cast(self, src, dst, casting):
        assert not np.can_cast(QuadPrecDType(backend=src), QuadPrecDType(backend=dst),
                               casting=casting)


@pytest.mark.parametrize("src,dst", PAIRS)
class TestScalarsOfDifferentBackends:
    def test_item_assignment(self, src, dst):
        target = _arr([7, 7], dst)
        with _raises():
            target[0] = QuadPrecision("1.5", backend=src)
        np.testing.assert_array_equal(target, _arr([7, 7], dst))

    def test_array_from_scalars(self, src, dst):
        # Without a common instance NumPy keeps the scalars in an object array.
        scalars = [QuadPrecision("1.5", backend=src), QuadPrecision("2.5", backend=dst)]
        objects = np.array(scalars)
        assert objects.dtype == np.dtype(object)
        assert all(o is s for o, s in zip(objects, scalars))
        with _raises():
            np.array([QuadPrecision(1, backend=src)], dtype=QuadPrecDType(backend=dst))
        with _raises():
            np.full(2, QuadPrecision("1.5", backend=src), dtype=QuadPrecDType(backend=dst))

    def test_object_array_cast(self, src, dst):
        objects = np.array([QuadPrecision("1.5", backend=src)], dtype=object)
        with _raises():
            objects.astype(QuadPrecDType(backend=dst))

    def test_constructor(self, src, dst):
        with _raises():
            QuadPrecision(QuadPrecision("1.5", backend=src), backend=dst)

    @pytest.mark.parametrize("op", [
        lambda x, y: x + y, lambda x, y: x * y, lambda x, y: x == y, lambda x, y: x < y,
    ])
    def test_scalar_ops(self, src, dst, op):
        with _raises():
            op(QuadPrecision(1, backend=src), QuadPrecision(2, backend=dst))


def test_constructor_default_backend_rejects_longdouble():
    with _raises():
        QuadPrecision(QuadPrecision("1.5", backend="longdouble"))


@pytest.mark.parametrize("op", [
    lambda a: a + 1.0, lambda a: 2 * a, lambda a: a > 0, lambda a: np.maximum(a, 1),
])
def test_python_literal_with_longdouble_array_raises(op):
    # NumPy gives Python int and float operands the default (SLEEF) descriptor.
    with pytest.raises(TypeError, match="Python int and float operands use 'sleef'"):
        op(_arr([1.0, 2.0], "longdouble"))


@pytest.mark.parametrize("func", [
    lambda dt, a: np.ones(3, dtype=dt),
    lambda dt, a: np.full(3, 1.5, dtype=dt),
    lambda dt, a: np.ones_like(a),
    lambda dt, a: a.mean(),
    lambda dt, a: a.std(),
    lambda dt, a: np.median(a),
    lambda dt, a: np.isclose(a, a),
    lambda dt, a: np.round(a, 1),
    lambda dt, a: a.var(),
    lambda dt, a: np.allclose(a, a),
], ids=["ones", "full", "ones_like", "mean", "std", "median", "isclose", "round", "var",
        "allclose"])
def test_numpy_functions_with_internal_python_scalars(func):
    ld = QuadPrecDType(backend="longdouble")
    with _raises():
        func(ld, _arr([1.0, 2.0, 3.0], "longdouble"))
    sleef = QuadPrecDType(backend="sleef")
    func(sleef, _arr([1.0, 2.0, 3.0], "sleef"))


def test_longdouble_workarounds():
    dt = QuadPrecDType(backend="longdouble")
    a = _arr([1.0, 2.0, 3.0], "longdouble")
    np.testing.assert_array_equal(np.full(3, np.longdouble(1), dtype=dt), _arr([1, 1, 1], "longdouble"))
    np.testing.assert_array_equal(np.ones(3).astype(dt), _arr([1, 1, 1], "longdouble"))
    np.testing.assert_array_equal(a / np.float64(2), _arr([0.5, 1.0, 1.5], "longdouble"))
    assert a.sum() / np.float64(a.size) == QuadPrecision(2, backend="longdouble")
    where = np.where(a > np.float64(1), a, 0)
    np.testing.assert_array_equal(where, _arr([0.0, 2.0, 3.0], "longdouble"))


@pytest.mark.parametrize("backend", BACKENDS)
class TestStillSupported:
    @pytest.mark.parametrize("other", [
        np.array([2.0, 4.0]),
        np.array([2.0, 4.0], dtype=np.float32),
        np.array([2, 4], dtype=np.int64),
        np.array([2.0, 4.0], dtype=np.longdouble),
    ], ids=["float64", "float32", "int64", "longdouble"])
    @pytest.mark.parametrize("quad_first", [True, False])
    def test_numpy_array_operand(self, backend, other, quad_first):
        quad = _arr([1.0, 3.0], backend)
        x, y = (quad, other) if quad_first else (other, quad)
        result = np.add(x, y)
        assert result.dtype == QuadPrecDType(backend=backend)
        np.testing.assert_array_equal(result, _arr([3.0, 7.0], backend))
        np.testing.assert_array_equal(x < y, quad < other if quad_first else other < quad)

    @pytest.mark.parametrize("scalar", [
        np.float64(2.0), np.float32(2.0), np.int32(2), np.longdouble(2.0),
    ], ids=["float64", "float32", "int32", "longdouble"])
    def test_numpy_scalar_operand(self, backend, scalar):
        result = _arr([1.0, 3.0], backend) * scalar
        assert result.dtype == QuadPrecDType(backend=backend)
        np.testing.assert_array_equal(result, _arr([2.0, 6.0], backend))

    def test_numpy_longdouble_casts(self, backend):
        values = np.array([1, 2, 3], dtype=np.longdouble) / np.longdouble(4)
        quad = values.astype(QuadPrecDType(backend=backend))
        assert quad.dtype == QuadPrecDType(backend=backend)
        back = quad.astype(np.longdouble)
        assert back.dtype == np.longdouble
        np.testing.assert_array_equal(back, values)

    def test_scalar_with_python_literal(self, backend):
        q = QuadPrecision("1.5", backend=backend)
        assert q + 1.0 == QuadPrecision("2.5", backend=backend)
        assert np.asarray(q * 2).dtype == QuadPrecDType(backend=backend)
        assert q > 1

    def test_same_backend_exchange(self, backend):
        dt = QuadPrecDType(backend=backend)
        a = _arr([1, 2], backend)
        assert np.result_type(dt, dt) == dt
        assert np.concatenate([a, a]).dtype == dt
        target = _arr([0, 0], backend)
        target[...] = a
        target[0] = QuadPrecision(5, backend=backend)
        np.copyto(target, a[::-1])
        np.testing.assert_array_equal(target, _arr([2, 1], backend))
        assert np.sqrt(a, out=_arr([0, 0], backend)).dtype == dt

    def test_constructor_copies_exactly(self, backend):
        tiny = QuadPrecision(2.0, backend=backend) ** -60
        value = QuadPrecision(1, backend=backend) + tiny
        assert value != QuadPrecision(1, backend=backend)
        copy = QuadPrecision(value, backend=backend)
        assert copy == value
        assert copy.as_integer_ratio() == value.as_integer_ratio()


def test_sleef_array_with_python_literals():
    a = _arr([1.0, 2.0], "sleef")
    np.testing.assert_array_equal(a + 1.0, _arr([2.0, 3.0], "sleef"))
    np.testing.assert_array_equal(2 * a, _arr([2.0, 4.0], "sleef"))
    np.testing.assert_array_equal(a > 1, [False, True])


def test_numpy_dtypes_unaffected():
    ld = np.array([1, 2], dtype=np.longdouble)
    assert (ld + 1.0).dtype == np.longdouble
    assert np.result_type(np.longdouble, np.float64) == np.longdouble
    assert np.concatenate([ld, np.array([3.0])]).dtype == np.longdouble
