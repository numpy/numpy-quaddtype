#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL QuadPrecType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_4_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC

extern "C" {
#include <Python.h>
#include <cstdio>
#include <string.h>

#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/dtype_api.h"
}

#include <climits>

#include "quad_common.h"
#include "scalar.h"
#include "dtype.h"
#include "ops.hpp"
#include "umath/matmul.h"
#include "umath/promoters.hpp"
#include "quadblas_interface.h"

// Mixed backends stay unsupported until inter-backend casts are lossless (gh-112, gh-113).
static NPY_CASTING
quad_dot_family_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                    PyArray_DTypeMeta *const NPY_UNUSED(dtypes[]),
                                    PyArray_Descr *const given_descrs[],
                                    PyArray_Descr *loop_descrs[],
                                    npy_intp *NPY_UNUSED(view_offset))
{
    for (int i = 0; i < 3; i++) {
        loop_descrs[i] = NULL;
    }

    QuadBackendType backend = ((QuadPrecDTypeObject *)given_descrs[0])->backend;
    if (((QuadPrecDTypeObject *)given_descrs[1])->backend != backend ||
        (given_descrs[2] != NULL &&
         ((QuadPrecDTypeObject *)given_descrs[2])->backend != backend)) {
        PyErr_SetString(PyExc_NotImplementedError,
                        "QuadPrecision dot products require all operands to use the same "
                        "backend");
        return quad_resolve_descrs_fail(loop_descrs, 3);
    }

    for (int i = 0; i < 2; i++) {
        Py_INCREF(given_descrs[i]);
        loop_descrs[i] = given_descrs[i];
    }
    if (given_descrs[2] == NULL) {
        loop_descrs[2] = (PyArray_Descr *)new_quaddtype_instance(backend);
        if (loop_descrs[2] == NULL) {
            return quad_resolve_descrs_fail(loop_descrs, 3);
        }
    }
    else {
        Py_INCREF(given_descrs[2]);
        loop_descrs[2] = given_descrs[2];
    }
    return NPY_NO_CASTING;
}

enum MatmulOperationType {
    MATMUL_DOT,
    MATMUL_GEMV,
    MATMUL_GEMM
};

static MatmulOperationType
determine_operation_type(npy_intp m, npy_intp n, npy_intp p)
{
    if (m == 1 && p == 1) {
        return MATMUL_DOT;
    }
    else if (p == 1) {
        return MATMUL_GEMV;
    }
    else {
        return MATMUL_GEMM;
    }
}

static int
naive_matmul_strided_loop(PyArrayMethod_Context *context, char *const data[],
                          npy_intp const dimensions[], npy_intp const strides[],
                          NpyAuxData *auxdata);

static inline int
matmul_stride_ok(npy_intp s)
{
    npy_intp elsize = (npy_intp)sizeof(Sleef_quad);
    return s > 0 && (s % elsize == 0) && (s / elsize) <= INT_MAX;
}

static inline int
is_blasable2d(npy_intp s_slow, npy_intp s_fast, npy_intp d_slow, npy_intp d_fast)
{
    npy_intp elsize = (npy_intp)sizeof(Sleef_quad);
    if (s_fast != elsize) {
        return 0;
    }
    npy_intp unit_slow = s_slow / elsize;
    return (s_slow % elsize == 0) && (unit_slow >= d_fast) && (unit_slow <= INT_MAX);
}

static int
matmul_alloc_count(npy_intp a, npy_intp b, size_t *out)
{
    if (a < 0 || b < 0) {
        return -1;
    }
    if (a != 0 && b > NPY_MAX_INTP / a) {
        return -1;
    }
    npy_intp count = a * b;
    if ((size_t)count > ((size_t)-1) / sizeof(Sleef_quad)) {
        return -1;
    }
    *out = (size_t)count;
    return 0;
}

static void
matmul_gather(Sleef_quad *dst, const char *src, npy_intp rows, npy_intp cols,
              npy_intp row_stride, npy_intp col_stride)
{
    for (npy_intp i = 0; i < rows; i++) {
        const char *row = src + i * row_stride;
        for (npy_intp j = 0; j < cols; j++) {
            memcpy(&dst[i * cols + j], row + j * col_stride, sizeof(Sleef_quad));
        }
    }
}

static void
matmul_scatter(char *dst, const Sleef_quad *src, npy_intp rows, npy_intp cols,
               npy_intp row_stride, npy_intp col_stride)
{
    for (npy_intp i = 0; i < rows; i++) {
        char *row = dst + i * row_stride;
        for (npy_intp j = 0; j < cols; j++) {
            memcpy(row + j * col_stride, &src[i * cols + j], sizeof(Sleef_quad));
        }
    }
}

// Inner loops may run without the GIL.
static int
quad_loop_fail(PyObject *exc, const char *msg)
{
    PyGILState_STATE gil = PyGILState_Ensure();
    PyErr_SetString(exc, msg);
    PyGILState_Release(gil);
    return -1;
}

static void
quad_dot_naive(char *ip1, npy_intp is1, char *ip2, npy_intp is2, char *op, npy_intp n,
               QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        Sleef_quad sum = Sleef_cast_from_doubleq1(0.0);
        for (npy_intp k = 0; k < n; k++, ip1 += is1, ip2 += is2) {
            Sleef_quad a, b;
            memcpy(&a, ip1, sizeof(Sleef_quad));
            memcpy(&b, ip2, sizeof(Sleef_quad));
            sum = Sleef_fmaq1_u05(a, b, sum);
        }
        memcpy(op, &sum, sizeof(Sleef_quad));
    }
    else {
        long double sum = 0.0L;
        for (npy_intp k = 0; k < n; k++, ip1 += is1, ip2 += is2) {
            long double a, b;
            memcpy(&a, ip1, sizeof(long double));
            memcpy(&b, ip2, sizeof(long double));
            sum += a * b;
        }
        quad_longdouble_store(op, sum);
    }
}

// Requires aligned operands. QBLAS is SLEEF-only, so longdouble always takes the
// strided loop; route it through QBLAS here once QBLAS supports it.
static int
quad_dot(char *ip1, npy_intp is1, char *ip2, npy_intp is2, char *op, npy_intp n,
         QuadBackendType backend)
{
#ifndef DISABLE_QUADBLAS
    if (backend == BACKEND_SLEEF && n <= INT_MAX && matmul_stride_ok(is1) &&
        matmul_stride_ok(is2)) {
        const npy_intp elsize = (npy_intp)sizeof(Sleef_quad);
        if (qblas_dot((size_t)n, (Sleef_quad *)ip1, (size_t)(is1 / elsize), (Sleef_quad *)ip2,
                      (size_t)(is2 / elsize), (Sleef_quad *)op) != 0) {
            return quad_loop_fail(PyExc_RuntimeError, "QBLAS operation failed");
        }
        return 0;
    }
#endif
    quad_dot_naive(ip1, is1, ip2, is2, op, n, backend);
    return 0;
}

static int
quad_matmul_strided_loop_aligned(PyArrayMethod_Context *context, char *const data[],
                                 npy_intp const dimensions[], npy_intp const strides[],
                                 NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    npy_intp m = dimensions[1];
    npy_intp n = dimensions[2];
    npy_intp p = dimensions[3];

    npy_intp A_stride = strides[0];
    npy_intp B_stride = strides[1];
    npy_intp C_stride = strides[2];
    npy_intp A_row_stride = strides[3];
    npy_intp A_col_stride = strides[4];
    npy_intp B_row_stride = strides[5];
    npy_intp B_col_stride = strides[6];
    npy_intp C_row_stride = strides[7];
    npy_intp C_col_stride = strides[8];

    // QBLAS is SLEEF-only; longdouble takes the strided loop until QBLAS supports it.
    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    if (descr->backend != BACKEND_SLEEF || m > INT_MAX || n > INT_MAX || p > INT_MAX ||
        m == 0 || n == 0 || p == 0) {
        return naive_matmul_strided_loop(context, data, dimensions, strides, auxdata);
    }

    MatmulOperationType op_type = determine_operation_type(m, n, p);
    Sleef_quad alpha = Sleef_cast_from_doubleq1(1.0);
    Sleef_quad beta = Sleef_cast_from_doubleq1(0.0);
    const npy_intp elsize = (npy_intp)sizeof(Sleef_quad);

    char *A_base = data[0];
    char *B_base = data[1];
    char *C_base = data[2];

    if (op_type == MATMUL_DOT) {
        for (npy_intp batch = 0; batch < N; batch++) {
            if (quad_dot(A_base + batch * A_stride, A_col_stride, B_base + batch * B_stride,
                         B_row_stride, C_base + batch * C_stride, n, BACKEND_SLEEF) < 0) {
                return -1;
            }
        }
        return 0;
    }

    if (op_type == MATMUL_GEMV) {
        char transa;
        size_t lda, gemv_rows, gemv_cols;
        int A_copy = 0;
        if (is_blasable2d(A_row_stride, A_col_stride, m, n)) {
            transa = 'N';
            gemv_rows = (size_t)m;
            gemv_cols = (size_t)n;
            lda = (size_t)(A_row_stride / elsize);
        }
        else if (is_blasable2d(A_col_stride, A_row_stride, n, m)) {
            transa = 'T';
            gemv_rows = (size_t)n;
            gemv_cols = (size_t)m;
            lda = (size_t)(A_col_stride / elsize);
        }
        else {
            transa = 'N';
            gemv_rows = (size_t)m;
            gemv_cols = (size_t)n;
            lda = (size_t)n;
            A_copy = 1;
        }

        int x_copy = !matmul_stride_ok(B_row_stride);
        int y_copy = !matmul_stride_ok(C_row_stride);
        size_t incx = x_copy ? 1 : (size_t)(B_row_stride / elsize);
        size_t incy = y_copy ? 1 : (size_t)(C_row_stride / elsize);

        size_t cntA = 0;
        if (A_copy && matmul_alloc_count(m, n, &cntA) != 0) {
            return naive_matmul_strided_loop(context, data, dimensions, strides, auxdata);
        }

        Sleef_quad *tA = NULL, *tx = NULL, *ty = NULL;
        if (A_copy) tA = (Sleef_quad *)PyMem_RawMalloc(cntA * sizeof(Sleef_quad));
        if (x_copy) tx = (Sleef_quad *)PyMem_RawMalloc((size_t)n * sizeof(Sleef_quad));
        if (y_copy) ty = (Sleef_quad *)PyMem_RawMalloc((size_t)m * sizeof(Sleef_quad));
        if ((A_copy && !tA) || (x_copy && !tx) || (y_copy && !ty)) {
            PyMem_RawFree(tA);
            PyMem_RawFree(tx);
            PyMem_RawFree(ty);
            PyErr_NoMemory();
            return -1;
        }

        int result = 0;
        for (npy_intp batch = 0; batch < N; batch++) {
            char *A_b = A_base + batch * A_stride;
            char *B_b = B_base + batch * B_stride;
            char *C_b = C_base + batch * C_stride;
            Sleef_quad *A_use = (Sleef_quad *)A_b;
            Sleef_quad *x_use = (Sleef_quad *)B_b;
            Sleef_quad *y_use = (Sleef_quad *)C_b;
            if (A_copy) {
                matmul_gather(tA, A_b, m, n, A_row_stride, A_col_stride);
                A_use = tA;
            }
            if (x_copy) {
                matmul_gather(tx, B_b, n, 1, B_row_stride, elsize);
                x_use = tx;
            }
            if (y_copy) {
                y_use = ty;
            }
            result = qblas_gemv('R', transa, gemv_rows, gemv_cols, &alpha, A_use, lda,
                                x_use, incx, &beta, y_use, incy);
            if (result == 0 && y_copy) {
                matmul_scatter(C_b, ty, m, 1, C_row_stride, elsize);
            }
            if (result != 0) {
                break;
            }
        }
        PyMem_RawFree(tA);
        PyMem_RawFree(tx);
        PyMem_RawFree(ty);
        if (result != 0) {
            PyErr_SetString(PyExc_RuntimeError, "QBLAS operation failed");
            return -1;
        }
        return 0;
    }

    char transa, transb;
    size_t lda, ldb, ldc;
    int A_copy = 0, B_copy = 0, C_copy = 0;
    if (is_blasable2d(A_row_stride, A_col_stride, m, n)) {
        transa = 'N';
        lda = (size_t)(A_row_stride / elsize);
    }
    else if (is_blasable2d(A_col_stride, A_row_stride, n, m)) {
        transa = 'T';
        lda = (size_t)(A_col_stride / elsize);
    }
    else {
        transa = 'N';
        lda = (size_t)n;
        A_copy = 1;
    }
    if (is_blasable2d(B_row_stride, B_col_stride, n, p)) {
        transb = 'N';
        ldb = (size_t)(B_row_stride / elsize);
    }
    else if (is_blasable2d(B_col_stride, B_row_stride, p, n)) {
        transb = 'T';
        ldb = (size_t)(B_col_stride / elsize);
    }
    else {
        transb = 'N';
        ldb = (size_t)p;
        B_copy = 1;
    }
    if (is_blasable2d(C_row_stride, C_col_stride, m, p)) {
        ldc = (size_t)(C_row_stride / elsize);
    }
    else {
        ldc = (size_t)p;
        C_copy = 1;
    }

    size_t cntA = 0, cntB = 0, cntC = 0;
    if ((A_copy && matmul_alloc_count(m, n, &cntA) != 0) ||
        (B_copy && matmul_alloc_count(n, p, &cntB) != 0) ||
        (C_copy && matmul_alloc_count(m, p, &cntC) != 0)) {
        return naive_matmul_strided_loop(context, data, dimensions, strides, auxdata);
    }

    Sleef_quad *tA = NULL, *tB = NULL, *tC = NULL;
    if (A_copy) tA = (Sleef_quad *)PyMem_RawMalloc(cntA * sizeof(Sleef_quad));
    if (B_copy) tB = (Sleef_quad *)PyMem_RawMalloc(cntB * sizeof(Sleef_quad));
    if (C_copy) tC = (Sleef_quad *)PyMem_RawMalloc(cntC * sizeof(Sleef_quad));
    if ((A_copy && !tA) || (B_copy && !tB) || (C_copy && !tC)) {
        PyMem_RawFree(tA);
        PyMem_RawFree(tB);
        PyMem_RawFree(tC);
        PyErr_NoMemory();
        return -1;
    }

    int result = 0;
    for (npy_intp batch = 0; batch < N; batch++) {
        char *A_b = A_base + batch * A_stride;
        char *B_b = B_base + batch * B_stride;
        char *C_b = C_base + batch * C_stride;
        Sleef_quad *A_use = (Sleef_quad *)A_b;
        Sleef_quad *B_use = (Sleef_quad *)B_b;
        Sleef_quad *C_use = (Sleef_quad *)C_b;
        if (A_copy) {
            matmul_gather(tA, A_b, m, n, A_row_stride, A_col_stride);
            A_use = tA;
        }
        if (B_copy) {
            matmul_gather(tB, B_b, n, p, B_row_stride, B_col_stride);
            B_use = tB;
        }
        if (C_copy) {
            C_use = tC;
        }
        result = qblas_gemm('R', transa, transb, (size_t)m, (size_t)p, (size_t)n,
                            &alpha, A_use, lda, B_use, ldb, &beta, C_use, ldc);
        if (result == 0 && C_copy) {
            matmul_scatter(C_b, tC, m, p, C_row_stride, C_col_stride);
        }
        if (result != 0) {
            break;
        }
    }
    PyMem_RawFree(tA);
    PyMem_RawFree(tB);
    PyMem_RawFree(tC);
    if (result != 0) {
        PyErr_SetString(PyExc_RuntimeError, "QBLAS operation failed");
        return -1;
    }
    return 0;
}

static int
naive_matmul_strided_loop(PyArrayMethod_Context *context, char *const data[],
                          npy_intp const dimensions[], npy_intp const strides[],
                          NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    npy_intp m = dimensions[1];
    npy_intp n = dimensions[2];
    npy_intp p = dimensions[3];

    npy_intp A_batch_stride = strides[0];
    npy_intp B_batch_stride = strides[1];
    npy_intp C_batch_stride = strides[2];

    npy_intp A_row_stride = strides[3];
    npy_intp A_col_stride = strides[4];
    npy_intp B_row_stride = strides[5];
    npy_intp B_col_stride = strides[6];
    npy_intp C_row_stride = strides[7];
    npy_intp C_col_stride = strides[8];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;

    char *A_base = data[0];
    char *B_base = data[1];
    char *C_base = data[2];

    for (npy_intp batch = 0; batch < N; batch++) {
        char *A = A_base + batch * A_batch_stride;
        char *B = B_base + batch * B_batch_stride;
        char *C = C_base + batch * C_batch_stride;

        for (npy_intp i = 0; i < m; i++) {
            for (npy_intp j = 0; j < p; j++) {
                quad_dot_naive(A + i * A_row_stride, A_col_stride, B + j * B_col_stride,
                               B_row_stride, C + i * C_row_stride + j * C_col_stride, n,
                               backend);
            }
        }
    }

    return 0;
}

static int
quad_vecdot_strided_loop(PyArrayMethod_Context *context, char *const data[],
                         npy_intp const dimensions[], npy_intp const strides[],
                         NpyAuxData *NPY_UNUSED(auxdata))
{
    QuadBackendType backend = ((QuadPrecDTypeObject *)context->descriptors[0])->backend;
    char *ip1 = data[0], *ip2 = data[1], *op = data[2];

    for (npy_intp i = 0; i < dimensions[0];
         i++, ip1 += strides[0], ip2 += strides[1], op += strides[2]) {
        if (quad_dot(ip1, strides[3], ip2, strides[4], op, dimensions[1], backend) < 0) {
            return -1;
        }
    }
    return 0;
}

// NumPy hands generalized ufunc loops aligned operands (copying unaligned ones),
// so no unaligned loop is registered.
static int
quad_add_dot_family_loop(PyObject *numpy, const char *ufunc_name, const char *spec_name,
                         PyArrayMethod_StridedLoop *loop)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }

    PyArray_DTypeMeta *dtypes[3] = {&QuadPrecDType, &QuadPrecDType, &QuadPrecDType};
    PyType_Slot slots[] = {
            {NPY_METH_resolve_descriptors, (void *)&quad_dot_family_resolve_descriptors},
            {NPY_METH_strided_loop, (void *)loop},
            {0, NULL}};
    PyArrayMethod_Spec spec = {
            .name = spec_name,
            .nin = 2,
            .nout = 1,
            .casting = NPY_NO_CASTING,
            .flags = (NPY_ARRAYMETHOD_FLAGS)0,
            .dtypes = dtypes,
            .slots = slots,
    };

    int res = PyUFunc_AddLoopFromSpec(ufunc, &spec);
    if (res == 0) {
        res = quad_add_promoters(ufunc);
    }
    Py_DECREF(ufunc);
    return res;
}

int
init_matmul_ops(PyObject *numpy)
{
#ifndef DISABLE_QUADBLAS
    return quad_add_dot_family_loop(numpy, "matmul", "quad_matmul_qblas",
                                    &quad_matmul_strided_loop_aligned);
#else
    return quad_add_dot_family_loop(numpy, "matmul", "quad_matmul_qblas",
                                    &naive_matmul_strided_loop);
#endif  // DISABLE_QUADBLAS
}

int
init_vecdot_ops(PyObject *numpy)
{
    return quad_add_dot_family_loop(numpy, "vecdot", "quad_vecdot", &quad_vecdot_strided_loop);
}
