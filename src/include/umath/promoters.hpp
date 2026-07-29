#ifndef _QUADDTYPE_PROMOTERS
#define _QUADDTYPE_PROMOTERS

#include <Python.h>
#include <cstdio>
#include <cassert>
#include "numpy/arrayobject.h"
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/dtype_api.h"

#include "../dtype.h"

inline bool
quad_ufunc_has_object_input(PyUFuncObject *ufunc, PyArray_DTypeMeta *const op_dtypes[])
{
    for (int i = 0; i < ufunc->nin; i++) {
        if (op_dtypes[i] == &PyArray_ObjectDType) {
            return true;
        }
    }
    return false;
}

inline void
quad_set_promoted_dtype(PyArray_DTypeMeta *signature_dtype,
                        PyArray_DTypeMeta *fallback_dtype,
                        PyArray_DTypeMeta **new_dtype)
{
    PyArray_DTypeMeta *dtype = signature_dtype != NULL ? signature_dtype : fallback_dtype;
    Py_INCREF(dtype);
    *new_dtype = dtype;
}

inline int
quad_ufunc_promoter(PyObject *ufunc_obj, PyArray_DTypeMeta *const op_dtypes[],
                    PyArray_DTypeMeta *const signature[], PyArray_DTypeMeta *new_op_dtypes[])
{
    PyUFuncObject *ufunc = (PyUFuncObject *)ufunc_obj;
    int nargs = ufunc->nargs;

    // Handle the special case for reductions
    if (op_dtypes[0] == NULL) {
        assert(ufunc->nin == 2 && ufunc->nout == 1); /* must be reduction */
        for (int i = 0; i < 3; i++) {
            Py_INCREF(op_dtypes[1]);
            new_op_dtypes[i] = op_dtypes[1];
        }
        return 0;
    }

    if (quad_ufunc_has_object_input(ufunc, op_dtypes)) {
        for (int i = 0; i < nargs; i++) {
            quad_set_promoted_dtype(signature[i], &PyArray_ObjectDType, &new_op_dtypes[i]);
        }
        return 0;
    }

    // This promoter is only registered for patterns where at least one
    // input is QuadPrecDType, so we always promote all args to QuadPrecDType.
    for (int i = 0; i < nargs; i++) {
        quad_set_promoted_dtype(signature[i], &QuadPrecDType, &new_op_dtypes[i]);
    }

    return 0;
}


inline int
quad_ldexp_promoter(PyObject *ufunc_obj, PyArray_DTypeMeta *const op_dtypes[],
                    PyArray_DTypeMeta *const signature[], PyArray_DTypeMeta *new_op_dtypes[])
{
    Py_INCREF(&QuadPrecDType);
    new_op_dtypes[0] = &QuadPrecDType;

    // Promote the exponent to PyArray_IntpDType (unless signature specifies otherwise)
    if (signature[1] != NULL) {
        Py_INCREF(signature[1]);
        new_op_dtypes[1] = signature[1];
    }
    else {
        Py_INCREF(&PyArray_IntpDType);
        new_op_dtypes[1] = &PyArray_IntpDType;
    }

    Py_INCREF(&QuadPrecDType);
    new_op_dtypes[2] = &QuadPrecDType;

    return 0;
}

#endif