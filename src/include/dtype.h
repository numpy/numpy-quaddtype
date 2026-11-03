#ifndef _QUADDTYPE_DTYPE_H
#define _QUADDTYPE_DTYPE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/dtype_api.h>
#include "quad_common.h"

typedef struct {
    PyArray_Descr base;
    QuadBackendType backend;
} QuadPrecDTypeObject;

extern PyArray_DTypeMeta QuadPrecDType;

QuadPrecDTypeObject *
new_quaddtype_instance(QuadBackendType backend);

/* Fail a resolver; `loop_descrs` must be NULL-initialized by the caller. */
static inline NPY_CASTING
quad_resolve_descrs_fail(PyArray_Descr *loop_descrs[], int n)
{
    for (int i = 0; i < n; i++) {
        Py_CLEAR(loop_descrs[i]);
    }
    return (NPY_CASTING)-1;
}

/*
 * Values never move between backends until inter-backend casts are exact and
 * a promotion policy is defined (gh-112, gh-113).
 */
static inline void
quad_set_backend_mismatch_error(void)
{
    PyErr_SetString(PyExc_TypeError,
                    "Cannot mix QuadPrecision backends 'sleef' and 'longdouble' "
                    "(Python int and float operands use 'sleef')");
}

/*
 * Backend for a loop over Quad operands. NumPy hands non-Quad operands to the
 * resolver as default (SLEEF) descriptors, so a longdouble operand decides;
 * genuinely mixed Quad operands are then rejected by the Quad-to-Quad cast.
 */
static inline QuadBackendType
quad_loop_backend(PyArray_Descr *const descrs[], int n)
{
    for (int i = 0; i < n; i++) {
        if (((QuadPrecDTypeObject *)descrs[i])->backend == BACKEND_LONGDOUBLE) {
            return BACKEND_LONGDOUBLE;
        }
    }
    return BACKEND_SLEEF;
}

/*
 * Loop descriptor for a Quad output given input descriptor `in`. An `out` of
 * another backend still gets `in`'s backend, so NumPy's cast into it raises.
 */
static inline PyArray_Descr *
quad_out_loop_descr(PyArray_Descr *in, PyArray_Descr *out)
{
    PyArray_Descr *res = in;
    if (out != NULL &&
        ((QuadPrecDTypeObject *)out)->backend == ((QuadPrecDTypeObject *)in)->backend) {
        res = out;
    }
    Py_INCREF(res);
    return res;
}

int
init_quadprec_dtype(void);

#ifdef __cplusplus
}
#endif

#endif