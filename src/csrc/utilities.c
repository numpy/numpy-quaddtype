#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "utilities.h"
#include "constants.hpp"

int
ascii_isspace(int c)
{
    return c == ' ' || c == '\f' || c == '\n' || c == '\r' || c == '\t' || c == '\v';
}

int
ascii_isalpha(char c)
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

int
ascii_isdigit(char c)
{
    return (c >= '0' && c <= '9');
}

int
ascii_isalnum(char c)
{
    return ascii_isdigit(c) || ascii_isalpha(c);
}

int
ascii_tolower(int c)
{
    if (c >= 'A' && c <= 'Z') {
        return c + ('a' - 'A');
    }
    return c;
}

// inspired from NumPyOS_ascii_strncasecmp
int
ascii_strncasecmp(const char *s1, const char *s2, size_t n)
{
    while (n > 0 && *s1 != '\0' && *s2 != '\0') {
        int c1 = ascii_tolower((unsigned char)*s1);
        int c2 = ascii_tolower((unsigned char)*s2);
        int diff = c1 - c2;
        
        if (diff != 0) {
            return diff;
        }

        s1++;
        s2++;
        n--;
    }

    if(n > 0) {
        return *s1 - *s2;
    }
    return 0;
}

/*
 * Internal helper: Parse numeric string to quad-precision.
 * Assumes no leading whitespace and no inf/nan (caller handles those).
 * The 'start' parameter points to where the original string started (for endptr on error).
 */
static int
cstring_to_quad_internal(const char *str, const char *start, QuadBackendType backend, 
                         quad_value *out_value, char **endptr)
{
    if (backend == BACKEND_SLEEF) {
        // SLEEF 4.0's Sleef_strtoq doesn't properly set endptr to indicate
        // where parsing stopped. We need to manually validate and track the parse position.
        
        const char *p = str;
        
        // Handle optional sign
        if (*p == '+' || *p == '-') {
            p++;
        }
        
        // Must have at least one digit or decimal point followed by digit
        int has_digits = 0;
        
        // Parse integer part
        while (ascii_isdigit(*p)) {
            has_digits = 1;
            p++;
        }
        
        // Parse decimal point and fractional part
        if (*p == '.') {
            p++;
            while (ascii_isdigit(*p)) {
                has_digits = 1;
                p++;
            }
        }
        
        // Must have at least one digit somewhere
        if (!has_digits) {
            if (endptr) *endptr = (char *)start;
            return -1;
        }
        
        // Parse optional exponent
        if (*p == 'e' || *p == 'E') {
            const char *exp_start = p;
            p++;
            
            // Optional sign in exponent
            if (*p == '+' || *p == '-') {
                p++;
            }
            
            // Must have at least one digit in exponent
            if (!ascii_isdigit(*p)) {
                // Invalid exponent, backtrack
                p = exp_start;
            } else {
                while (ascii_isdigit(*p)) {
                    p++;
                }
            }
        }
        
        // Now p points to where valid parsing ends
        // Create a null-terminated substring for SLEEF
        size_t len = p - str;
        char *temp = (char *)malloc(len + 1);
        if (!temp) {
            if (endptr) *endptr = (char *)start;
            return -1;
        }
        memcpy(temp, str, len);
        temp[len] = '\0';
        
        // Call Sleef_strtoq with the bounded string
        //
        // DEVELOPER NOTE - Sleef_strtoq decimal precision limit.
        // The decimal branch of Sleef_strtoq is NOT correctly-rounded. It
        // accumulates digits (n = n*10 + d) into SLEEF's triple-double "tdx"
        // format (a 64-bit exponent + vdouble3 mantissa, ~159 mantissa bits),
        // scales by exp10i() and rounds via cast_vq_tdx(). Only the hex (0x...)
        // branch is exact bit-manipulation.
        //
        // Three different digit counts are involved - do NOT conflate them:
        //   * ~34 = a quad's actual precision              (113 * log10(2))
        //   * 36  = digits that GUARANTEE any quad round-trips; this is
        //           SLEEF_QUAD_DECIMAL_DIG and the most Dragon4 (str/repr) emits
        //   * ~48 = the tdx mantissa's raw capacity         (159 bits / log2(10))
        //   * ~45 = the EFFECTIVE window this parser resolves correctly - a few
        //           digits below 48 because the tdx add/mul/div/exp10i are not
        //           perfectly rounded. Property of the parser, not of quads.
        //
        // ~45 is NOT a hard/coded limit: it is a soft, build-dependent threshold,
        // and it is measured (empirical), not read from a constant. Sweeping
        // genuine quads across the exponent range, the smallest k at which
        // nudging a half-ULP midpoint by 10^-k gets silently dropped (rounds the
        // wrong way) was 45 (some values tolerate 46-48). Inputs longer than the
        // window are still read - the extra digits just stop reliably affecting
        // the result. So if the digit that decides rounding sits at position
        // >= ~46 - e.g. a value just above a midpoint - the result can be 1 ULP
        // off. Concrete repro
        // (SLEEF 3.9.0, x86): the ~46-digit-deep value near
        // "1.99999...9997111...e0" parses to ...ffffe, not the correct ...fffff.
        //
        // Practical impact on quad<->string round-trips: none here. Every quad
        // is captured by <= 36 digits, ~9 below the ~45 window, so Dragon4
        // output re-parses exactly. BUT that ~45 is FMA-/codegen-dependent (the
        // tdx arithmetic can lose a guard bit on some toolchains, e.g. macOS)
        // and can shrink toward 36 - so do not rely on string parsing for
        // exactness. Binary paths (raw bytes) are always exact.
        char *sleef_endptr;
        out_value->sleef_value = Sleef_strtoq(temp, &sleef_endptr);
        free(temp);
        
        // Set endptr to our calculated position
        if (endptr) {
            *endptr = (char *)p;
        }
        
    } else {
        out_value->longdouble_value = strtold(str, endptr);
    }
    
    if (endptr && *endptr == str) {
        // Nothing was parsed - set endptr to original start
        *endptr = (char *)start;
        return -1;
    }
    
    return 0; // success
}

/*
 * NumPyOS_ascii_strtoq:
 *
 * Locale-independent string to quad-precision parser.
 * Inspired by NumPyOS_ascii_strtold from NumPy.
 *
 * This function:
 * - Skips leading whitespace
 * - Recognizes inf/nan case-insensitively with optional signs and payloads
 * - Parses numeric values
 *
 * Returns:
 *   0 on success
 *  -1 on parse error
 */
int
NumPyOS_ascii_strtoq(const char *s, QuadBackendType backend, quad_value *out_value, char **endptr)
{
    const char *p;
    int sign;
    
    // skip leading whitespace
    while (ascii_isspace(*s)) {
        s++;
    }
    
    p = s;
    sign = 1;
    if (*p == '-') {
        sign = -1;
        ++p;
    }
    else if (*p == '+') {
        ++p;
    }

    // Check for inf/infinity (case-insensitive)
    if (ascii_strncasecmp(p, "inf", 3) == 0) {
        p += 3;
        if (ascii_strncasecmp(p, "inity", 5) == 0) {
            p += 5;
        }

        // Set infinity values with sign applied
        if (backend == BACKEND_SLEEF) {
            out_value->sleef_value = sign > 0 ? QUAD_PRECISION_INF : QUAD_PRECISION_NINF;
        }
        else {
            out_value->longdouble_value = sign > 0 ? strtold("inf", NULL) : strtold("-inf", NULL);
        }
        
        if (endptr) {
            *endptr = (char *)p;
        }
        return 0;
    }

    // Check for nan (case-insensitive) with optional payload
    if (ascii_strncasecmp(p, "nan", 3) == 0) {
        p += 3;

        // Skip optional (payload)
        if (*p == '(') {
            ++p;
            while (ascii_isalnum(*p) || *p == '_') {
                ++p;
            }
            if (*p == ')') {
                ++p;
            }
        }

        // Set NaN value with sign preserved
        if (backend == BACKEND_SLEEF) {
            Sleef_quad nan_val = QUAD_PRECISION_NAN;
            // Apply sign to NaN (negative NaN has sign bit set)
            if (sign < 0) {
                nan_val = Sleef_negq1(nan_val);
            }
            out_value->sleef_value = nan_val;
        }
        else {
            out_value->longdouble_value = sign < 0 ? -nanl("") : nanl("");
        }
        
        if (endptr) {
            *endptr = (char *)p;
        }
        return 0;
    }
    
    // For numeric values, parse starting from 's' (includes sign if present)
    // The sign is part of the number, not handled separately like inf/nan
    return cstring_to_quad_internal(s, s, backend, out_value, endptr);
}

// Helper function: Convert quad_value to Sleef_quad for Dragon4
Sleef_quad
quad_to_sleef_quad(const quad_value *in_val, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        // can directly return but that causes union heisenbugs, 
        // but this helper is rare to use, so acceptable
        Sleef_quad result;
        memcpy(&result, &in_val->sleef_value, sizeof(Sleef_quad));
        return result;
    }
    else {
        return Sleef_cast_from_doubleq1((double)(in_val->longdouble_value));
    }
}