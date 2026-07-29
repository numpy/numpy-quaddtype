#ifndef _QUADDTYPE_COMMON_H
#define _QUADDTYPE_COMMON_H

#include <float.h>
#include <stddef.h>
#include <string.h>

#include <sleef.h>
#include <sleefquad.h>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define X87_LONG_DOUBLE_VALUE_BYTES 10

typedef enum {
    BACKEND_INVALID = -1,
    BACKEND_SLEEF,
    BACKEND_LONGDOUBLE
} QuadBackendType;

typedef union {
    Sleef_quad sleef_value;
    long double longdouble_value;
} quad_value;

static inline void
quad_value_zero(quad_value *value)
{
    // Make all object-representation bytes observable before assigning a value.
#if defined(__GNUC__) || defined(__clang__)
    memset(value, 0, sizeof(*value));
    __asm__ __volatile__("" : : "r"(value) : "memory");
#elif defined(_MSC_VER)
    memset(value, 0, sizeof(*value));
    _ReadWriteBarrier();
#else
    volatile unsigned char *bytes = (volatile unsigned char *)value;
    for (size_t i = 0; i < sizeof(*value); i++) {
        bytes[i] = 0;
    }
#endif
}

static inline void
quad_longdouble_store(void *dst, long double value)
{
#if (defined(__i386__) || defined(__x86_64__)) && LDBL_MANT_DIG == 64 && LDBL_MAX_EXP == 16384
    // x87 extended precision occupies bytes 0..9; the remaining bytes are padding.
    memset(dst, 0, sizeof(value));
    memcpy(dst, &value, X87_LONG_DOUBLE_VALUE_BYTES);
#elif LDBL_MANT_DIG == 64 && LDBL_MAX_EXP == 16384
    quad_value canonical;
    quad_value_zero(&canonical);
    canonical.longdouble_value = value;
    memcpy(dst, &canonical, sizeof(canonical.longdouble_value));
#else
    memcpy(dst, &value, sizeof(value));
#endif
}

static inline void
quad_longdouble_store_aligned(void *dst, long double value)
{
#if (defined(__i386__) || defined(__x86_64__)) && LDBL_MANT_DIG == 64 && LDBL_MAX_EXP == 16384
    *(long double *)dst = value;
    memset((unsigned char *)dst + X87_LONG_DOUBLE_VALUE_BYTES, 0,
           sizeof(value) - X87_LONG_DOUBLE_VALUE_BYTES);
#else
    quad_longdouble_store(dst, value);
#endif
}

static inline void
quad_value_set_longdouble(quad_value *value, long double input)
{
    quad_longdouble_store_aligned(&value->longdouble_value, input);
}

static inline void
quad_value_load(quad_value *value, const void *src, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        memcpy(&value->sleef_value, src, sizeof(value->sleef_value));
    }
    else {
        memcpy(&value->longdouble_value, src, sizeof(value->longdouble_value));
    }
}

static inline void
quad_value_load_canonical(quad_value *value, const void *src, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        memcpy(&value->sleef_value, src, sizeof(value->sleef_value));
    }
    else {
        long double input;
        memcpy(&input, src, sizeof(input));
        quad_value_set_longdouble(value, input);
    }
}

static inline void
quad_value_store(void *dst, const quad_value *value, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        memcpy(dst, &value->sleef_value, sizeof(value->sleef_value));
    }
    else {
        quad_longdouble_store(dst, value->longdouble_value);
    }
}

static inline void
quad_value_store_aligned(void *dst, const quad_value *value, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        memcpy(dst, &value->sleef_value, sizeof(value->sleef_value));
    }
    else {
        quad_longdouble_store_aligned(dst, value->longdouble_value);
    }
}

// For IEEE 754 binary128 (quad precision), we need 36 decimal digits 
// to guarantee round-trip conversion (string -> parse -> equals original value)
// Formula: ceil(1 + MANT_DIG * log10(2)) = ceil(1 + 113 * 0.30103) = 36
// src: https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format
#define SLEEF_QUAD_DECIMAL_DIG 36

#ifdef __cplusplus
}
#endif

#endif