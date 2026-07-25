import contextlib
import ctypes
import platform
import random

import numpy as np
import pytest

from numpy_quaddtype import QuadPrecDType


SLEEF = QuadPrecDType(backend="sleef")
LONGDOUBLE = QuadPrecDType(backend="longdouble")
LD_INFO = np.finfo(np.longdouble)
LD_PRECISION = LD_INFO.nmant + 1

BEYOND_DOUBLE = [
    "9007199254740993",
    "1.000000000000000000867361737988403547205962240695953369140625",
]
SPECIAL_VALUES = ["0", "-0", "inf", "-inf", "nan", "-nan"]
BACKEND_CASTS = [
    pytest.param("longdouble", "sleef", "safe", id="longdouble-to-sleef"),
    pytest.param("sleef", "longdouble", "same_kind", id="sleef-to-longdouble"),
]
ROUNDING_MODES = [
    pytest.param("nearest", id="nearest-even"),
    pytest.param("down", id="downward"),
    pytest.param("up", id="upward"),
    pytest.param("zero", id="toward-zero"),
]


def _dtype(backend):
    return QuadPrecDType(backend=backend)


def _as_longdouble(value, target=LONGDOUBLE):
    return value.astype(target).astype(np.longdouble)[0]


def _as_float64(value):
    return value.astype(np.float64)[0]


def _assert_value_and_sign(actual, expected):
    assert actual == expected
    if actual == 0:
        assert np.signbit(actual) == np.signbit(expected)


def _require_narrower_longdouble():
    if LD_INFO.nmant >= 112:
        pytest.skip("long double has the same precision as SLEEF binary128")


@contextlib.contextmanager
def _rounding_mode(name):
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Windows":
        control = ctypes.CDLL("msvcrt")._controlfp_s
        current = ctypes.c_uint()
        control(ctypes.byref(current), 0, 0)
        values = {"nearest": 0, "down": 0x100, "up": 0x200, "zero": 0x300}
        control(ctypes.byref(ctypes.c_uint()), values[name], 0x300)
        try:
            yield
        finally:
            control(ctypes.byref(ctypes.c_uint()), current.value, 0x300)
        return

    if system not in {"Linux", "Darwin"} or machine not in {
        "x86_64",
        "amd64",
        "aarch64",
        "arm64",
    }:
        pytest.skip("rounding-mode constants are unknown on this platform")

    libc = ctypes.CDLL(None)
    if not hasattr(libc, "fegetround") or not hasattr(libc, "fesetround"):
        pytest.skip("floating-point environment API is unavailable")
    libc.fegetround.restype = ctypes.c_int
    libc.fesetround.argtypes = [ctypes.c_int]
    libc.fesetround.restype = ctypes.c_int
    if machine in {"aarch64", "arm64"}:
        values = {
            "nearest": 0,
            "down": 0x800000,
            "up": 0x400000,
            "zero": 0xC00000,
        }
    else:
        values = {"nearest": 0, "down": 0x400, "up": 0x800, "zero": 0xC00}
    previous = libc.fegetround()
    assert libc.fesetround(values[name]) == 0
    try:
        yield
    finally:
        assert libc.fesetround(previous) == 0


@pytest.mark.parametrize("value", BEYOND_DOUBLE)
@pytest.mark.parametrize("source_kind", ["quad", "numpy"])
def test_longdouble_to_sleef_preserves_values_beyond_double(value, source_kind):
    if LD_INFO.nmant <= np.finfo(np.float64).nmant:
        pytest.skip("long double has no precision beyond double")

    source = (
        np.array([value], dtype=LONGDOUBLE)
        if source_kind == "quad"
        else np.array([np.longdouble(value)], dtype=np.longdouble)
    )
    expected = np.array([value], dtype=SLEEF)

    result = source.astype(SLEEF, casting="safe")

    np.testing.assert_array_equal(result, expected, strict=True)


@pytest.mark.parametrize("value", BEYOND_DOUBLE)
def test_longdouble_to_sleef_same_value_beyond_double(value):
    if LD_INFO.nmant <= np.finfo(np.float64).nmant:
        pytest.skip("long double has no precision beyond double")

    source = np.array([value], dtype=LONGDOUBLE)
    expected = np.array([value], dtype=SLEEF)

    result = source.astype(SLEEF, casting="same_value")

    np.testing.assert_array_equal(result, expected, strict=True)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "1.000000000000000000867361737988403547205962240695953369140625",
            "1.000000000000000000867361737988403547205962240695953369140625",
        ),
        (
            "1.0000000000000000000542101086242752217003726400434970855712890625",
            "1.0",
        ),
        (
            "1.0000000000000000001626303258728256651011179201304912567138671875",
            "1.00000000000000000021684043449710088680149056017398834228515625",
        ),
    ],
)
@pytest.mark.parametrize("target_kind", ["quad", "numpy"])
def test_sleef_to_x87_rounds_once_to_nearest_even(source, expected, target_kind):
    if LD_INFO.nmant != 63:
        pytest.skip("test values target x87 64-bit significand rounding")

    value = np.array([source], dtype=SLEEF)
    if target_kind == "quad":
        result = value.astype(LONGDOUBLE).astype(np.longdouble)
    else:
        result = value.astype(np.longdouble)
    expected_array = np.array([np.longdouble(expected)], dtype=np.longdouble)

    np.testing.assert_array_equal(result, expected_array, strict=True)


@pytest.mark.parametrize("value", SPECIAL_VALUES)
@pytest.mark.parametrize(("source_backend", "target_backend", "_"), BACKEND_CASTS)
def test_inter_backend_special_values_preserve_class_and_sign(
    value, source_backend, target_backend, _
):
    source = np.array([value], dtype=_dtype(source_backend))

    result = source.astype(_dtype(target_backend))

    assert np.signbit(result[0]) == np.signbit(source[0])
    if "nan" in value:
        assert np.isnan(result[0])
    elif "inf" in value:
        assert np.isinf(result[0])
    else:
        assert float(result[0]) == float(source[0]) == 0.0


@pytest.mark.parametrize("casting", ["safe", "same_value"])
def test_longdouble_to_sleef_random_exact_roundtrip(casting):
    rng = random.Random(1847)
    values = [
        np.longdouble(0),
        np.longdouble("-0"),
        LD_INFO.max,
        -LD_INFO.max,
        LD_INFO.tiny,
        -LD_INFO.tiny,
        np.nextafter(np.longdouble(0), np.longdouble(1), dtype=np.longdouble),
    ]
    for _ in range(256):
        significand = (1 << (LD_PRECISION - 1)) | rng.getrandbits(LD_PRECISION - 1)
        exponent = rng.randint(LD_INFO.minexp, LD_INFO.maxexp - 1)
        value = np.ldexp(np.longdouble(str(significand)), exponent - LD_PRECISION)
        values.append(-value if rng.getrandbits(1) else value)

    original = np.array(values, dtype=np.longdouble)
    source = original.astype(LONGDOUBLE)
    widened = source.astype(SLEEF, casting=casting)
    roundtrip = widened.astype(LONGDOUBLE, casting="same_value")
    actual = roundtrip.astype(np.longdouble)

    np.testing.assert_array_equal(actual, original, strict=True)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(original))


@pytest.mark.parametrize("mode", ROUNDING_MODES)
def test_sleef_to_longdouble_rounding_modes(mode):
    _require_narrower_longdouble()
    one = np.array(["1"], dtype=SLEEF)
    next_up = np.nextafter(np.longdouble(1), np.longdouble(2), dtype=np.longdouble)
    next_down = np.nextafter(np.longdouble(-1), np.longdouble(-2), dtype=np.longdouble)
    upper = np.array([next_up], dtype=np.longdouble).astype(LONGDOUBLE).astype(SLEEF)
    lower_negative = (
        np.array([next_down], dtype=np.longdouble).astype(LONGDOUBLE).astype(SLEEF)
    )
    positive_midpoint = (one + upper) / np.array(["2"], dtype=SLEEF)
    negative_midpoint = (-one + lower_negative) / np.array(["2"], dtype=SLEEF)
    expected = {
        "nearest": (np.longdouble(1), np.longdouble(-1)),
        "down": (np.longdouble(1), next_down),
        "up": (next_up, np.longdouble(-1)),
        "zero": (np.longdouble(1), np.longdouble(-1)),
    }

    with _rounding_mode(mode):
        actual = (_as_longdouble(positive_midpoint), _as_longdouble(negative_midpoint))

    for result, wanted in zip(actual, expected[mode], strict=True):
        _assert_value_and_sign(result, wanted)


@pytest.mark.parametrize("mode", ROUNDING_MODES)
def test_sleef_to_longdouble_underflow_and_overflow_modes(mode):
    _require_narrower_longdouble()
    one = np.array(["1"], dtype=SLEEF)
    zero = np.longdouble(0)
    min_subnormal = np.nextafter(zero, np.longdouble(1), dtype=np.longdouble)
    half_min_subnormal = np.ldexp(one, LD_INFO.minexp - LD_PRECISION)
    max_value = (
        np.array([LD_INFO.max], dtype=np.longdouble).astype(LONGDOUBLE).astype(SLEEF)
    )
    overflow_midpoint = max_value + np.ldexp(one, LD_INFO.maxexp - LD_PRECISION - 1)
    expected = {
        "nearest": (zero, -zero, np.longdouble("inf"), np.longdouble("-inf")),
        "down": (zero, -min_subnormal, LD_INFO.max, np.longdouble("-inf")),
        "up": (min_subnormal, -zero, np.longdouble("inf"), -LD_INFO.max),
        "zero": (zero, -zero, LD_INFO.max, -LD_INFO.max),
    }

    with _rounding_mode(mode), np.errstate(over="ignore", under="ignore"):
        actual = (
            _as_longdouble(half_min_subnormal),
            _as_longdouble(-half_min_subnormal),
            _as_longdouble(overflow_midpoint),
            _as_longdouble(-overflow_midpoint),
        )

    for result, wanted in zip(actual, expected[mode], strict=True):
        _assert_value_and_sign(result, wanted)


@pytest.mark.parametrize("mode", ROUNDING_MODES)
def test_sleef_to_float64_rounding_modes(mode):
    one = np.array(["1"], dtype=SLEEF)
    next_up = np.nextafter(np.float64(1), np.float64(2))
    next_down = np.nextafter(np.float64(-1), np.float64(-2))
    upper = np.array([next_up], dtype=np.float64).astype(SLEEF)
    lower_negative = np.array([next_down], dtype=np.float64).astype(SLEEF)
    positive_midpoint = (one + upper) / np.array(["2"], dtype=SLEEF)
    negative_midpoint = (-one + lower_negative) / np.array(["2"], dtype=SLEEF)
    expected = {
        "nearest": (np.float64(1), np.float64(-1)),
        "down": (np.float64(1), next_down),
        "up": (next_up, np.float64(-1)),
        "zero": (np.float64(1), np.float64(-1)),
    }

    with _rounding_mode(mode):
        actual = (_as_float64(positive_midpoint), _as_float64(negative_midpoint))

    for result, wanted in zip(actual, expected[mode], strict=True):
        _assert_value_and_sign(result, wanted)


@pytest.mark.parametrize("mode", ROUNDING_MODES)
def test_sleef_to_float64_underflow_and_overflow_modes(mode):
    info = np.finfo(np.float64)
    precision = info.nmant + 1
    one = np.array(["1"], dtype=SLEEF)
    zero = np.float64(0)
    min_subnormal = np.nextafter(zero, np.float64(1))
    half_min_subnormal = np.ldexp(one, info.minexp - precision)
    max_value = np.array([info.max], dtype=np.float64).astype(SLEEF)
    overflow_midpoint = max_value + np.ldexp(one, info.maxexp - precision - 1)
    far_overflow = np.array(["1e4000"], dtype=SLEEF)
    expected = {
        "nearest": (
            zero,
            -zero,
            np.float64("inf"),
            np.float64("-inf"),
            np.float64("inf"),
            np.float64("-inf"),
        ),
        "down": (
            zero,
            -min_subnormal,
            info.max,
            np.float64("-inf"),
            info.max,
            np.float64("-inf"),
        ),
        "up": (
            min_subnormal,
            -zero,
            np.float64("inf"),
            -info.max,
            np.float64("inf"),
            -info.max,
        ),
        "zero": (zero, -zero, info.max, -info.max, info.max, -info.max),
    }

    with _rounding_mode(mode), np.errstate(over="ignore", under="ignore"):
        actual = (
            _as_float64(half_min_subnormal),
            _as_float64(-half_min_subnormal),
            _as_float64(overflow_midpoint),
            _as_float64(-overflow_midpoint),
            _as_float64(far_overflow),
            _as_float64(-far_overflow),
        )

    for result, wanted in zip(actual, expected[mode], strict=True):
        _assert_value_and_sign(result, wanted)


@pytest.mark.parametrize("exponent", [-1020, -100, -1, 0, 1, 100, 1022])
@pytest.mark.parametrize("sign", [1, -1])
def test_sleef_to_float64_adjacent_midpoints_across_exponents(exponent, sign):
    value = np.ldexp(np.float64("0.75"), exponent)
    value = value if sign > 0 else -value
    direction = np.float64("inf") if sign > 0 else np.float64("-inf")
    neighbor = np.nextafter(value, direction)
    lower = np.array([value], dtype=np.float64).astype(SLEEF)
    upper = np.array([neighbor], dtype=np.float64).astype(SLEEF)
    midpoint = (lower + upper) / np.array(["2"], dtype=SLEEF)

    assert _as_float64(np.nextafter(midpoint, lower)) == value
    assert _as_float64(np.nextafter(midpoint, upper)) == neighbor


@pytest.mark.parametrize(
    "exponent",
    [
        LD_INFO.minexp + 2,
        -1000,
        -1,
        0,
        1,
        1000,
        LD_INFO.maxexp - 2,
    ],
)
@pytest.mark.parametrize("sign", [1, -1])
def test_sleef_to_longdouble_adjacent_midpoints_across_exponents(exponent, sign):
    _require_narrower_longdouble()
    value = np.ldexp(np.longdouble("0.75"), exponent)
    value = value if sign > 0 else -value
    direction = np.longdouble("inf") if sign > 0 else np.longdouble("-inf")
    neighbor = np.nextafter(value, direction, dtype=np.longdouble)
    lower = np.array([value], dtype=np.longdouble).astype(LONGDOUBLE).astype(SLEEF)
    upper = np.array([neighbor], dtype=np.longdouble).astype(LONGDOUBLE).astype(SLEEF)
    midpoint = (lower + upper) / np.array(["2"], dtype=SLEEF)

    toward_value = np.nextafter(midpoint, lower)
    toward_neighbor = np.nextafter(midpoint, upper)

    assert _as_longdouble(toward_value) == value
    assert _as_longdouble(toward_neighbor) == neighbor


@pytest.mark.parametrize(("source_backend", "target_backend", "casting"), BACKEND_CASTS)
@pytest.mark.parametrize("shape", [(0,), (17,), (3, 4), (2, 3, 5)])
@pytest.mark.parametrize("layout", ["contiguous", "reversed", "unaligned"])
def test_inter_backend_casts_support_shapes_and_strides(
    source_backend, target_backend, casting, shape, layout
):
    source_dtype = _dtype(source_backend)
    target_dtype = _dtype(target_backend)
    count = int(np.prod(shape))
    values = np.linspace(-100, 100, count, dtype=np.longdouble).reshape(shape)
    aligned = values.astype(source_dtype)

    if layout == "contiguous":
        source = aligned
        expected_source = aligned
        result = source.astype(target_dtype, casting=casting)
    elif layout == "reversed":
        source = aligned[..., ::-1]
        expected_source = source
        result = source.astype(target_dtype, casting=casting)
    else:
        source = np.ndarray(
            shape,
            dtype=source_dtype,
            buffer=bytearray(1 + count * source_dtype.itemsize),
            offset=1,
        )
        target = np.ndarray(
            shape,
            dtype=target_dtype,
            buffer=bytearray(1 + count * target_dtype.itemsize),
            offset=1,
        )
        source[...] = aligned
        expected_source = aligned
        np.copyto(target, source, casting=casting)
        result = target

    expected = expected_source.astype(target_dtype, casting=casting)
    np.testing.assert_array_equal(result, expected, strict=True)
