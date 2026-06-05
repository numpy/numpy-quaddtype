// numpy-quaddtype shim around QBLAS.

#include "quadblas_interface.h"
#include <cstring>
#include <algorithm>

#ifndef DISABLE_QUADBLAS
#include <qblas/qblas.h>
#endif

extern "C" {

#ifndef DISABLE_QUADBLAS

static inline QBLAS_LAYOUT to_layout(char c) {
    return (c == 'C' || c == 'c') ? QblasColMajor : QblasRowMajor;
}
static inline QBLAS_TRANSPOSE to_trans(char c) {
    if (c == 'T' || c == 't') return QblasTrans;
    if (c == 'C' || c == 'c') return QblasConjTrans;
    return QblasNoTrans;
}

int
qblas_dot(size_t n, Sleef_quad *x, size_t incx,
          Sleef_quad *y, size_t incy, Sleef_quad *result)
{
    if (!result) {
        return -1;
    }
    if (n == 0) {
        *result = Sleef_cast_from_doubleq1(0.0);
        return 0;
    }
    if (!x || !y) {
        return -1;
    }
    *result = cblas_qdot((int)n, x, (int)incx, y, (int)incy);
    return 0;
}

int
qblas_gemv(char layout, char trans, size_t m, size_t n,
           Sleef_quad *alpha, Sleef_quad *A, size_t lda,
           Sleef_quad *x, size_t incx,
           Sleef_quad *beta, Sleef_quad *y, size_t incy)
{
    if (m == 0 || n == 0) {
        if (y && beta) {
            int do_trans = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');
            size_t y_len = do_trans ? n : m;
            Sleef_quad zero = Sleef_cast_from_doubleq1(0.0);
            int beta_zero = Sleef_icmpeqq1(*beta, zero);
            for (size_t i = 0; i < y_len; i++) {
                y[i * incy] = beta_zero ? zero : Sleef_mulq1_u05(*beta, y[i * incy]);
            }
        }
        return 0;
    }
    if (!alpha || !A || !x || !beta || !y) {
        return -1;
    }
    cblas_qgemv(to_layout(layout), to_trans(trans),
                (int)m, (int)n,
                *alpha, A, (int)lda,
                x, (int)incx,
                *beta, y, (int)incy);
    return 0;
}

int
qblas_gemm(char layout, char transa, char transb,
           size_t m, size_t n, size_t k,
           Sleef_quad *alpha, Sleef_quad *A, size_t lda,
           Sleef_quad *B, size_t ldb,
           Sleef_quad *beta, Sleef_quad *C, size_t ldc)
{
    if (m == 0 || n == 0) {
        return 0;
    }
    if (!alpha || !A || !B || !beta || !C) {
        return -1;
    }
    cblas_qgemm(to_layout(layout), to_trans(transa), to_trans(transb),
                (int)m, (int)n, (int)k,
                *alpha, A, (int)lda, B, (int)ldb,
                *beta,  C, (int)ldc);
    return 0;
}

int
qblas_supports_backend(QuadBackendType backend)
{
    return (backend == BACKEND_SLEEF) ? 1 : 0;
}

PyObject *
py_quadblas_set_num_threads(PyObject *self, PyObject *args)
{
    int num_threads;
    if (!PyArg_ParseTuple(args, "i", &num_threads)) {
        return NULL;
    }
    if (num_threads <= 0) {
        PyErr_SetString(PyExc_ValueError, "Number of threads must be positive");
        return NULL;
    }
    qblas_set_num_threads(num_threads);
    Py_RETURN_NONE;
}

PyObject *
py_quadblas_get_num_threads(PyObject *self, PyObject *args)
{
    return PyLong_FromLong(qblas_get_num_threads());
}

PyObject *
py_quadblas_get_version(PyObject *self, PyObject *args)
{
    /* qblas_get_version() returns "QBLAS X.Y.Z"; pair it with the
     * runtime-detected SIMD tier so callers can confirm what's active. */
    const char *ver  = qblas_get_version();
    const char *tier = qblas_get_dispatch_tier();
    char buf[256];
    PyOS_snprintf(buf, sizeof buf, "%s (dispatch: %s)", ver, tier);
    return PyUnicode_FromString(buf);
}

int
_quadblas_set_num_threads(int num_threads)
{
    qblas_set_num_threads(num_threads);
    return 0;
}

int
_quadblas_get_num_threads(void)
{
    return qblas_get_num_threads();
}

#else  /* DISABLE_QUADBLAS */

int
qblas_dot(size_t n, Sleef_quad *x, size_t incx, Sleef_quad *y, size_t incy, Sleef_quad *result)
{ return -1; }

int
qblas_gemv(char layout, char trans, size_t m, size_t n, Sleef_quad *alpha, Sleef_quad *A,
           size_t lda, Sleef_quad *x, size_t incx, Sleef_quad *beta, Sleef_quad *y, size_t incy)
{ return -1; }

int
qblas_gemm(char layout, char transa, char transb, size_t m, size_t n, size_t k, Sleef_quad *alpha,
           Sleef_quad *A, size_t lda, Sleef_quad *B, size_t ldb, Sleef_quad *beta, Sleef_quad *C,
           size_t ldc)
{ return -1; }

int qblas_supports_backend(QuadBackendType backend) { return -1; }

PyObject *
py_quadblas_set_num_threads(PyObject *self, PyObject *args)
{
    PyErr_SetString(PyExc_NotImplementedError, "QuadBLAS is disabled");
    return NULL;
}
PyObject *
py_quadblas_get_num_threads(PyObject *self, PyObject *args)
{
    PyErr_SetString(PyExc_NotImplementedError, "QuadBLAS is disabled");
    return NULL;
}
PyObject *
py_quadblas_get_version(PyObject *self, PyObject *args)
{
    PyErr_SetString(PyExc_NotImplementedError, "QuadBLAS is disabled");
    return NULL;
}

int _quadblas_set_num_threads(int num_threads)
{
    PyErr_SetString(PyExc_NotImplementedError, "QuadBLAS is disabled");
    return -1;
}
int _quadblas_get_num_threads(void)
{
    PyErr_SetString(PyExc_NotImplementedError, "QuadBLAS is disabled");
    return -1;
}

#endif /* DISABLE_QUADBLAS */

}  /* extern "C" */
