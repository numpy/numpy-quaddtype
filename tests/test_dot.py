import pytest
import numpy as np
from utils import create_quad_array, assert_quad_equal, assert_quad_array_equal, arrays_equal_with_nan, _qnd
from numpy_quaddtype import QuadPrecision, QuadPrecDType


# ================================================================================
# VECTOR-VECTOR DOT PRODUCT TESTS
# ================================================================================

class TestVectorVectorDot:
    """Test vector-vector np.matmul products"""
    
    def test_simple_dot_product(self):
        """Test basic vector np.matmul product"""
        x = create_quad_array([1, 2, 3])
        y = create_quad_array([4, 5, 6])
        
        result = np.matmul(x, y)
        expected = 1*4 + 2*5 + 3*6  # = 32
        
        assert isinstance(result, QuadPrecision)
        assert_quad_equal(result, expected)
    
    def test_orthogonal_vectors(self):
        """Test orthogonal vectors (should give zero)"""
        x = create_quad_array([1, 0, 0])
        y = create_quad_array([0, 1, 0])
        
        result = np.matmul(x, y)
        assert_quad_equal(result, 0.0)
    
    def test_same_vector(self):
        """Test np.matmul product of vector with itself"""
        x = create_quad_array([2, 3, 4])
        
        result = np.matmul(x, x)
        expected = 2*2 + 3*3 + 4*4  # = 29
        
        assert_quad_equal(result, expected)
    
    @pytest.mark.parametrize("size", [1, 2, 5, 10, 50, 100])
    def test_various_vector_sizes(self, size):
        """Test different vector sizes from small to large"""
        # Create vectors with known pattern
        x_vals = [i + 1 for i in range(size)]  # [1, 2, 3, ...]
        y_vals = [2 * (i + 1) for i in range(size)]  # [2, 4, 6, ...]
        
        x = create_quad_array(x_vals)
        y = create_quad_array(y_vals)
        
        result = np.matmul(x, y)
        expected = sum(x_vals[i] * y_vals[i] for i in range(size))
        
        assert_quad_equal(result, expected)
    
    def test_negative_and_fractional_values(self):
        """Test vectors with negative and fractional values"""
        x = create_quad_array([1.5, -2.5, 3.25])
        y = create_quad_array([-1.25, 2.75, -3.5])
        
        result = np.matmul(x, y)
        expected = 1.5*(-1.25) + (-2.5)*2.75 + 3.25*(-3.5)
        
        assert_quad_equal(result, expected)


# ================================================================================
# MATRIX-VECTOR MULTIPLICATION TESTS  
# ================================================================================

class TestMatrixVectorDot:
    """Test matrix-vector multiplication"""
    
    def test_simple_matrix_vector(self):
        """Test basic matrix-vector multiplication"""
        # 2x3 matrix
        A = create_quad_array([1, 2, 3, 4, 5, 6], shape=(2, 3))
        # 3x1 vector  
        x = create_quad_array([1, 1, 1])
        
        result = np.matmul(A, x)
        expected = [1+2+3, 4+5+6]  # [6, 15]
        
        assert result.shape == (2,)
        for i in range(2):
            assert_quad_equal(result[i], expected[i])
    
    def test_identity_matrix_vector(self):
        """Test multiplication with identity matrix"""
        # 3x3 identity matrix
        I = create_quad_array([1, 0, 0, 0, 1, 0, 0, 0, 1], shape=(3, 3))
        x = create_quad_array([2, 3, 4])
        
        result = np.matmul(I, x)
        
        assert result.shape == (3,)
        for i in range(3):
            assert_quad_equal(result[i], float(x[i]))
    
    @pytest.mark.parametrize("m,n", [(2,3), (3,2), (5,4), (10,8), (20,15)])
    def test_various_matrix_vector_sizes(self, m, n):
        """Test various matrix-vector sizes from small to large"""
        # Create m×n matrix with sequential values
        A_vals = [(i*n + j + 1) for i in range(m) for j in range(n)]
        A = create_quad_array(A_vals, shape=(m, n))
        
        # Create n×1 vector with simple values
        x_vals = [i + 1 for i in range(n)]
        x = create_quad_array(x_vals)
        
        result = np.matmul(A, x)
        
        assert result.shape == (m,)
        
        # Verify manually for small matrices
        if m <= 5 and n <= 5:
            for i in range(m):
                expected = sum(A_vals[i*n + j] * x_vals[j] for j in range(n))
                assert_quad_equal(result[i], expected)


# ================================================================================
# MATRIX-MATRIX MULTIPLICATION TESTS
# ================================================================================

class TestMatrixMatrixDot:
    """Test matrix-matrix multiplication"""
    
    def test_simple_matrix_matrix(self):
        """Test basic matrix-matrix multiplication"""
        # 2x2 matrices
        A = create_quad_array([1, 2, 3, 4], shape=(2, 2))
        B = create_quad_array([5, 6, 7, 8], shape=(2, 2))
        
        result = np.matmul(A, B)
        
        # Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        expected = [[19, 22], [43, 50]]
        
        assert result.shape == (2, 2)
        for i in range(2):
            for j in range(2):
                assert_quad_equal(result[i, j], expected[i][j])
    
    def test_identity_matrix_multiplication(self):
        """Test multiplication with identity matrix"""
        A = create_quad_array([1, 2, 3, 4], shape=(2, 2))
        I = create_quad_array([1, 0, 0, 1], shape=(2, 2))
        
        # A * I should equal A
        result1 = np.matmul(A, I)
        assert_quad_array_equal(result1, A)
        
        # I * A should equal A  
        result2 = np.matmul(I, A)
        assert_quad_array_equal(result2, A)
    
    @pytest.mark.parametrize("m,n,k", [(2,2,2), (2,3,4), (3,2,5), (4,4,4), (5,6,7)])
    def test_various_matrix_sizes(self, m, n, k):
        """Test various matrix sizes: (m×k) × (k×n) = (m×n)"""
        # Create A: m×k matrix
        A_vals = [(i*k + j + 1) for i in range(m) for j in range(k)]
        A = create_quad_array(A_vals, shape=(m, k))
        
        # Create B: k×n matrix  
        B_vals = [(i*n + j + 1) for i in range(k) for j in range(n)]
        B = create_quad_array(B_vals, shape=(k, n))
        
        result = np.matmul(A, B)
        
        assert result.shape == (m, n)
        
        # Verify manually for small matrices
        if m <= 3 and n <= 3 and k <= 3:
            for i in range(m):
                for j in range(n):
                    expected = sum(A_vals[i*k + l] * B_vals[l*n + j] for l in range(k))
                    assert_quad_equal(result[i, j], expected)
    
    def test_associativity(self):
        """Test matrix multiplication associativity: (A*B)*C = A*(B*C)"""
        # Use small 2x2 matrices for simplicity
        A = create_quad_array([1, 2, 3, 4], shape=(2, 2))
        B = create_quad_array([2, 1, 1, 2], shape=(2, 2))
        C = create_quad_array([1, 1, 2, 1], shape=(2, 2))
        
        # Compute (A*B)*C
        AB = np.matmul(A, B)
        result1 = np.matmul(AB, C)
        
        # Compute A*(B*C)
        BC = np.matmul(B, C)
        result2 = np.matmul(A, BC)
        
        assert_quad_array_equal(result1, result2, rtol=1e-25)


# ================================================================================
# SPECIAL VALUES EDGE CASE TESTS
# ================================================================================

class TestSpecialValueEdgeCases:
    """Test matmul with special IEEE 754 values (NaN, inf, -0.0)"""
    
    @pytest.mark.parametrize("special_val", ["0.0", "-0.0", "inf", "-inf", "nan", "-nan"])
    def test_vector_with_special_values(self, special_val):
        """Test vectors containing special values"""
        # Create vectors with special values
        x = create_quad_array([1.0, float(special_val), 2.0])
        y = create_quad_array([3.0, 4.0, 5.0])
        
        result = np.matmul(x, y)
        
        # Compare with float64 reference
        x_float = np.array([1.0, float(special_val), 2.0], dtype=np.float64)
        y_float = np.array([3.0, 4.0, 5.0], dtype=np.float64)
        expected = np.matmul(x_float, y_float)
        
        # Handle special value comparisons
        if np.isnan(expected):
            assert np.isnan(float(result))
        elif np.isinf(expected):
            assert np.isinf(float(result))
            assert np.sign(float(result)) == np.sign(expected)
        else:
            assert_quad_equal(result, expected)
    
    @pytest.mark.parametrize("special_val", ["0.0", "-0.0", "inf", "-inf", "nan"])
    def test_matrix_vector_with_special_values(self, special_val):
        """Test matrix-vector multiplication with special values"""
        # Matrix with special value
        A = create_quad_array([1.0, float(special_val), 3.0, 4.0], shape=(2, 2))
        x = create_quad_array([2.0, 1.0])
        
        result = np.matmul(A, x)
        
        # Compare with float64 reference  
        A_float = np.array([[1.0, float(special_val)], [3.0, 4.0]], dtype=np.float64)
        x_float = np.array([2.0, 1.0], dtype=np.float64)
        expected = np.matmul(A_float, x_float)
        
        assert result.shape == expected.shape
        for i in range(len(expected)):
            if np.isnan(expected[i]):
                assert np.isnan(float(result[i]))
            elif np.isinf(expected[i]):
                assert np.isinf(float(result[i]))
                assert np.sign(float(result[i])) == np.sign(expected[i])
            else:
                assert_quad_equal(result[i], expected[i])
    
    @pytest.mark.parametrize("special_val", ["0.0", "-0.0", "inf", "-inf", "nan"])
    def test_matrix_matrix_with_special_values(self, special_val):
        """Test matrix-matrix multiplication with special values"""
        A = create_quad_array([1.0, 2.0, float(special_val), 4.0], shape=(2, 2))
        B = create_quad_array([5.0, 6.0, 7.0, 8.0], shape=(2, 2))
        
        result = np.matmul(A, B)
        
        # Compare with float64 reference
        A_float = np.array([[1.0, 2.0], [float(special_val), 4.0]], dtype=np.float64)
        B_float = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
        expected = np.matmul(A_float, B_float)
        
        assert result.shape == expected.shape
        assert arrays_equal_with_nan(result, expected)
    
    def test_all_nan_matrix(self):
        """Test matrices filled with NaN"""
        A = create_quad_array([float("nan")] * 4, shape=(2, 2))
        B = create_quad_array([1, 2, 3, 4], shape=(2, 2))
        
        result = np.matmul(A, B)
        
        # Result should be all NaN (NaN * anything = NaN)
        for i in range(2):
            for j in range(2):
                assert np.isnan(float(result[i, j]))
    
    def test_inf_times_zero_produces_nan(self):
        """Test that Inf * 0 correctly produces NaN per IEEE 754"""
        # Create a scenario where Inf * 0 occurs in matrix multiplication
        A = create_quad_array([float("inf"), 1.0], shape=(1, 2))
        B = create_quad_array([0.0, 1.0], shape=(2, 1))
        
        result = np.matmul(A, B)
        
        # Result should be inf*0 + 1*1 = NaN + 1 = NaN
        assert np.isnan(float(result[0, 0])), "Inf * 0 should produce NaN per IEEE 754"
    
    def test_nan_propagation(self):
        """Test that NaN properly propagates through matrix operations"""
        A = create_quad_array([1.0, float("nan"), 3.0, 4.0], shape=(2, 2))
        B = create_quad_array([1.0, 0.0, 0.0, 1.0], shape=(2, 2))  # Identity
        
        result = np.matmul(A, B)
        
        # C[0,0] = 1*1 + nan*0 = 1 + nan = nan (nan*0 = nan, not like inf*0)
        # C[0,1] = 1*0 + nan*1 = 0 + nan = nan  
        # C[1,0] = 3*1 + 4*0 = 3 + 0 = 3
        # C[1,1] = 3*0 + 4*1 = 0 + 4 = 4
        assert np.isnan(float(result[0, 0]))
        assert np.isnan(float(result[0, 1]))
        assert_quad_equal(result[1, 0], 3.0)
        assert_quad_equal(result[1, 1], 4.0)
    
    def test_zero_division_and_indeterminate_forms(self):
        """Test handling of indeterminate forms in matrix operations"""
        # Test various indeterminate forms that should produce NaN
        
        # Case: Inf - Inf form
        A = create_quad_array([float("inf"), float("inf")], shape=(1, 2))
        B = create_quad_array([1.0, -1.0], shape=(2, 1))
        
        result = np.matmul(A, B)
        
        # Result should be inf*1 + inf*(-1) = inf - inf = NaN
        assert np.isnan(float(result[0, 0])), "Inf - Inf should produce NaN per IEEE 754"
    
    def test_mixed_inf_values(self):
        """Test matrices with mixed infinite values"""
        # Use all-ones matrix to avoid Inf * 0 = NaN issues
        A = create_quad_array([float("inf"), 2, float("-inf"), 3], shape=(2, 2))
        B = create_quad_array([1, 1, 1, 1], shape=(2, 2))  # All ones to avoid Inf*0
        
        result = np.matmul(A, B)
        
        # C[0,0] = inf*1 + 2*1 = inf + 2 = inf
        # C[0,1] = inf*1 + 2*1 = inf + 2 = inf  
        # C[1,0] = -inf*1 + 3*1 = -inf + 3 = -inf
        # C[1,1] = -inf*1 + 3*1 = -inf + 3 = -inf
        assert np.isinf(float(result[0, 0])) and float(result[0, 0]) > 0
        assert np.isinf(float(result[0, 1])) and float(result[0, 1]) > 0
        assert np.isinf(float(result[1, 0])) and float(result[1, 0]) < 0  
        assert np.isinf(float(result[1, 1])) and float(result[1, 1]) < 0


# ================================================================================
# DEGENERATE AND EMPTY CASE TESTS
# ================================================================================

class TestDegenerateCases:
    """Test edge cases with degenerate dimensions"""
    
    def test_single_element_matrices(self):
        """Test 1x1 matrix operations"""
        A = create_quad_array([3.0], shape=(1, 1))
        B = create_quad_array([4.0], shape=(1, 1))
        
        result = np.matmul(A, B)
        
        assert result.shape == (1, 1)
        assert_quad_equal(result[0, 0], 12.0)
    
    def test_single_element_vector(self):
        """Test operations with single-element vectors"""
        x = create_quad_array([5.0])
        y = create_quad_array([7.0])
        
        result = np.matmul(x, y)
        
        assert isinstance(result, QuadPrecision)
        assert_quad_equal(result, 35.0)
    
    def test_very_tall_matrix(self):
        """Test very tall matrices (1000x1)"""
        size = 1000
        A = create_quad_array([1.0] * size, shape=(size, 1))
        B = create_quad_array([2.0], shape=(1, 1))
        
        result = np.matmul(A, B)
        
        assert result.shape == (size, 1)
        for i in range(min(10, size)):  # Check first 10 elements
            assert_quad_equal(result[i, 0], 2.0)
    
    def test_very_wide_matrix(self):
        """Test very wide matrices (1x1000)"""
        size = 1000
        A = create_quad_array([1.0], shape=(1, 1))  
        B = create_quad_array([3.0] * size, shape=(1, size))
        
        result = np.matmul(A, B)
        
        assert result.shape == (1, size)
        for i in range(min(10, size)):  # Check first 10 elements
            assert_quad_equal(result[0, i], 3.0)
    
    def test_zero_matrices(self):
        """Test matrices filled with zeros"""
        A = create_quad_array([0.0] * 9, shape=(3, 3))
        B = create_quad_array([1, 2, 3, 4, 5, 6, 7, 8, 9], shape=(3, 3))
        
        result = np.matmul(A, B)
        
        assert result.shape == (3, 3)
        for i in range(3):
            for j in range(3):
                assert_quad_equal(result[i, j], 0.0)
    
    def test_repeated_row_matrix(self):
        """Test matrices with repeated rows"""
        # Matrix with all rows the same
        A = create_quad_array([1, 2, 3] * 3, shape=(3, 3))  # Each row is [1, 2, 3]
        B = create_quad_array([1, 0, 0, 0, 1, 0, 0, 0, 1], shape=(3, 3))  # Identity
        
        result = np.matmul(A, B)
        
        # Result should have all rows equal to [1, 2, 3]
        for i in range(3):
            assert_quad_equal(result[i, 0], 1.0)
            assert_quad_equal(result[i, 1], 2.0)
            assert_quad_equal(result[i, 2], 3.0)
    
    def test_repeated_column_matrix(self):
        """Test matrices with repeated columns"""
        A = create_quad_array([1, 0, 0, 0, 1, 0, 0, 0, 1], shape=(3, 3))  # Identity
        B = create_quad_array([2, 2, 2, 3, 3, 3, 4, 4, 4], shape=(3, 3))  # Each column repeated
        result = np.matmul(A, B)
        # Result should be same as B (identity multiplication)
        assert_quad_array_equal(result, B)


class TestEmptyDimensions:
    """Test operations with empty dimensions (zero-length)"""
    dtype = QuadPrecDType(backend='sleef')

    @pytest.mark.parametrize("A_shape,B_shape", [
        ((0,),    (0,)),     # 1-D dot,    n=0       → scalar 0
        ((0, 3),  (3, 4)),   # GEMM,       m=0       → (0, 4)
        ((2, 0),  (0, 3)),   # GEMM,       k=0       → (2, 3) zeros
        ((2, 3),  (3, 0)),   # GEMM,       p=0       → (2, 0)
        ((3, 0),  (0,)),     # GEMV,       n=0       → (3,) zeros
        ((0, 3),  (3,)),     # GEMV,       m=0       → (0,)
    ], ids=lambda s: "x".join(str(d) for d in s))
    def test_empty_matmul_matches_float64(self, A_shape, B_shape):
        A_q = np.zeros(A_shape, dtype=self.dtype)
        B_q = np.zeros(B_shape, dtype=self.dtype)
        A_f = np.zeros(A_shape, dtype=np.float64)
        B_f = np.zeros(B_shape, dtype=np.float64)

        R_q = np.matmul(A_q, B_q)
        R_f = np.matmul(A_f, B_f)

        # 1-D dot returns a 0-d scalar QuadPrecision (no .shape attribute).
        q_shape = () if isinstance(R_q, QuadPrecision) else R_q.shape
        assert q_shape == R_f.shape, f"shape {q_shape} != float64 {R_f.shape}"
        assert R_q.dtype == self.dtype

        R_q_as_f = (np.float64(R_q) if isinstance(R_q, QuadPrecision)
                    else np.asarray(R_q).astype(np.float64))
        assert np.array_equal(R_q_as_f, R_f), f"values {R_q_as_f!r} != {R_f!r}"


class TestOutKeyword:
    """np.matmul with a pre-allocated `out=` array. The result must be
    written into `out` and `out` must be returned (numpy ufunc semantics)."""

    dtype = QuadPrecDType(backend='sleef')

    @pytest.mark.parametrize("A_shape,B_shape,out_shape", [
        ((2, 3),  (3, 4),  (2, 4)),    # GEMM
        ((4, 5),  (5,),    (4,)),      # GEMV
        ((1, 3),  (3, 1),  (1, 1)),    # 1x1 GEMM degenerate
    ], ids=lambda s: "x".join(str(d) for d in s) if s else "scalar")
    def test_out_writes_in_place(self, A_shape, B_shape, out_shape):
        rng = np.random.default_rng(seed=7)
        A_f = rng.standard_normal(A_shape)
        B_f = rng.standard_normal(B_shape)
        A_q = _qnd(A_f)
        B_q = _qnd(B_f)

        # Pre-fill `out` with a sentinel; matmul must overwrite every cell.
        out = np.full(out_shape, 999.0, dtype=self.dtype)
        result = np.matmul(A_q, B_q, out=out)

        assert result is out, "np.matmul should return the `out` array"
        assert out.dtype == self.dtype

        ref = np.matmul(A_f, B_f)
        out_as_f = np.asarray(out).astype(np.float64)
        # Sentinel must be gone everywhere — catches partial writes.
        assert not np.any(out_as_f == 999.0), "sentinel cells were not written"
        assert out_as_f.shape == ref.shape
        np.testing.assert_allclose(out_as_f, ref, rtol=1e-14, atol=1e-14)

    def test_out_wrong_shape_raises(self):
        """Mismatched `out` shape must raise (numpy enforces this, not us,
        but the resolver shouldn't crash trying)."""
        A = _qnd(np.ones((2, 3)))
        B = _qnd(np.ones((3, 4)))
        bad_out = np.zeros((2, 5), dtype=self.dtype)  # wrong cols
        with pytest.raises(ValueError):
            np.matmul(A, B, out=bad_out)


class TestMixedBackend:
    """matmul with backend-mismatched or non-SLEEF operands must fail cleanly
    rather than crash. The exception class is loose because M2 may change
    the longdouble path; the contract here is just 'raises, no segfault'."""

    @pytest.mark.parametrize("backend_a,backend_b", [
        ("sleef",      "longdouble"),
        ("longdouble", "sleef"),
    ])
    def test_mixed_backend_raises(self, backend_a, backend_b):
        A = np.zeros((2, 3), dtype=QuadPrecDType(backend=backend_a))
        B = np.zeros((3, 2), dtype=QuadPrecDType(backend=backend_b))
        with pytest.raises((TypeError, NotImplementedError)):
            np.matmul(A, B)


# ================================================================================
# NUMERICAL STABILITY AND PRECISION TESTS
# ================================================================================

class TestNumericalStability:
    """Test numerical stability with extreme values"""
    
    def test_very_large_values(self):
        """Test matrices with very large values"""
        large_val = 1e100
        A = create_quad_array([large_val, 1, 1, large_val], shape=(2, 2))
        B = create_quad_array([1, 0, 0, 1], shape=(2, 2))  # Identity
        
        result = np.matmul(A, B)
        
        # Should preserve large values without overflow
        assert_quad_equal(result[0, 0], large_val)
        assert_quad_equal(result[1, 1], large_val)
        assert not np.isinf(float(result[0, 0]))
        assert not np.isinf(float(result[1, 1]))
    
    def test_very_small_values(self):
        """Test matrices with very small values"""
        small_val = 1e-100
        A = create_quad_array([small_val, 0, 0, small_val], shape=(2, 2))
        B = create_quad_array([1, 0, 0, 1], shape=(2, 2))  # Identity
        
        result = np.matmul(A, B)
        
        # Should preserve small values without underflow
        assert_quad_equal(result[0, 0], small_val)
        assert_quad_equal(result[1, 1], small_val)
        assert float(result[0, 0]) != 0.0
        assert float(result[1, 1]) != 0.0
    
    def test_mixed_scale_values(self):
        """Test matrices with mixed magnitude values"""
        A = create_quad_array([1e100, 1e-100, 1e50, 1e-50], shape=(2, 2))
        B = create_quad_array([1, 0, 0, 1], shape=(2, 2))  # Identity
        
        result = np.matmul(A, B)
        
        # All values should be preserved accurately
        assert_quad_equal(result[0, 0], 1e100)
        assert_quad_equal(result[0, 1], 1e-100)
        assert_quad_equal(result[1, 0], 1e50)
        assert_quad_equal(result[1, 1], 1e-50)
    
    def test_precision_critical_case(self):
        """Test case that would lose precision in double"""
        # Create a case where large values cancel in the dot product
        # Vector: [1e20, 1.0, -1e20] dot [1, 0, 1] should equal 1.0
        x = create_quad_array([1e20, 1.0, -1e20])
        y = create_quad_array([1.0, 0.0, 1.0])
        
        result = np.matmul(x, y)
        
        # The result should be 1e20*1 + 1.0*0 + (-1e20)*1 = 1e20 - 1e20 = 0, but we want 1
        # Let me fix this: [1e20, 1.0, -1e20] dot [0, 1, 0] = 1.0
        x = create_quad_array([1e20, 1.0, -1e20])
        y = create_quad_array([0.0, 1.0, 0.0])
        
        result = np.matmul(x, y)
        
        # This would likely fail in double precision due to representation issues
        assert_quad_equal(result, 1.0, atol=1e-25)
    
    def test_condition_number_extreme(self):
        """Test matrices with extreme condition numbers"""
        # Nearly singular matrix (very small determinant)
        eps = 1e-50
        A = create_quad_array([1, 1, 1, 1+eps], shape=(2, 2))
        B = create_quad_array([1, 0, 0, 1], shape=(2, 2))
        
        result = np.matmul(A, B)
        
        # Result should be computed accurately
        assert_quad_equal(result[0, 0], 1.0)
        assert_quad_equal(result[0, 1], 1.0)
        assert_quad_equal(result[1, 0], 1.0)
        assert_quad_equal(result[1, 1], 1.0 + eps)
    
    def test_accumulation_precision(self):
        """Test precision in accumulation of many terms"""
        size = 100
        # Create vectors where each term contributes equally
        x_vals = [1.0 / size] * size
        y_vals = [1.0] * size
        
        x = create_quad_array(x_vals)
        y = create_quad_array(y_vals)
        
        result = np.matmul(x, y)
        
        # Result should be exactly 1.0 
        assert_quad_equal(result, 1.0, atol=1e-25)


# ================================================================================
# CROSS-VALIDATION TESTS
# ================================================================================

class TestCrossValidation:
    """Test consistency with float64 reference implementations"""
    
    @pytest.mark.parametrize("size", [2, 3, 5, 10])
    def test_consistency_with_float64_vectors(self, size):
        """Test vector operations consistency with float64"""
        # Use values well within float64 range
        x_vals = [i + 0.5 for i in range(size)]
        y_vals = [2 * i + 1.5 for i in range(size)]
        
        # QuadPrecision computation
        x_quad = create_quad_array(x_vals)
        y_quad = create_quad_array(y_vals)
        result_quad = np.matmul(x_quad, y_quad)
        
        # float64 reference
        x_float = np.array(x_vals, dtype=np.float64)
        y_float = np.array(y_vals, dtype=np.float64)
        result_float = np.matmul(x_float, y_float)
        
        # Results should match within float64 precision
        assert_quad_equal(result_quad, result_float, rtol=1e-14)
    
    @pytest.mark.parametrize("m,n,k", [(2,2,2), (3,3,3), (4,5,6)])
    def test_consistency_with_float64_matrices(self, m, n, k):
        """Test matrix operations consistency with float64"""
        # Create test matrices with float64-representable values
        A_vals = [(i + j + 1) * 0.25 for i in range(m) for j in range(k)]
        B_vals = [(i * 2 + j) * 0.125 for i in range(k) for j in range(n)]
        
        # QuadPrecision computation
        A_quad = create_quad_array(A_vals, shape=(m, k))
        B_quad = create_quad_array(B_vals, shape=(k, n))
        result_quad = np.matmul(A_quad, B_quad)
        
        # float64 reference
        A_float = np.array(A_vals, dtype=np.float64).reshape(m, k)
        B_float = np.array(B_vals, dtype=np.float64).reshape(k, n)
        result_float = np.matmul(A_float, B_float)
        
        # Results should match within float64 precision
        for i in range(m):
            for j in range(n):
                assert_quad_equal(result_quad[i, j], result_float[i, j], rtol=1e-14)
    
    def test_quad_precision_advantage(self):
        """Test cases where quad precision shows advantage over float64"""
        A = create_quad_array([1.0, 1e-30], shape=(1, 2))
        B = create_quad_array([1.0, 1.0], shape=(2, 1))
        
        result_quad = np.matmul(A, B)
        
        # The result should be 1.0 + 1e-30 = 1.0000000000000000000000000000001
        expected = 1.0 + 1e-30
        assert_quad_equal(result_quad[0, 0], expected, rtol=1e-25)
        
        # Verify that this value is actually different from 1.0 in quad precision
        diff = result_quad[0, 0] - 1.0
        assert abs(diff) > 0  # Should be non-zero in quad precision


# ================================================================================
# LARGE MATRIX TESTS
# ================================================================================

class TestLargeMatrices:
    """Test performance and correctness with larger matrices"""
    
    @pytest.mark.parametrize("size", [50, 100, 200])
    def test_large_square_matrices(self, size):
        """Test large square matrix multiplication"""
        # Create matrices with simple pattern for verification
        A_vals = [1.0 if i == j else 0.1 for i in range(size) for j in range(size)]  # Near-diagonal
        B_vals = [1.0] * (size * size)  # All ones
        
        A = create_quad_array(A_vals, shape=(size, size))
        B = create_quad_array(B_vals, shape=(size, size))
        
        result = np.matmul(A, B)
        
        assert result.shape == (size, size)
        
        # Each element = sum of a row in A = 1.0 + 0.1*(size-1)
        expected_value = 1.0 + 0.1 * (size - 1)
        
        # Check diagonal and off-diagonal elements
        assert_quad_equal(result[0, 0], expected_value, rtol=1e-15, atol=1e-15)
        if size > 1:
            assert_quad_equal(result[0, 1], expected_value, rtol=1e-15, atol=1e-15)
        
        # Additional verification: check a few more elements
        if size > 2:
            assert_quad_equal(result[1, 0], expected_value, rtol=1e-15, atol=1e-15)
            assert_quad_equal(result[size//2, size//2], expected_value, rtol=1e-15, atol=1e-15)
    
    def test_large_vector_operations(self):
        """Test large vector np.matmul products"""
        size = 1000
        
        # Create vectors with known sum
        x_vals = [1.0] * size
        y_vals = [2.0] * size
        
        x = create_quad_array(x_vals)
        y = create_quad_array(y_vals)
        
        result = np.matmul(x, y)
        expected = size * 1.0 * 2.0  # = 2000.0
        
        assert_quad_equal(result, expected)
    
    def test_rectangular_large_matrices(self):
        """Test large rectangular matrix operations"""
        m, n, k = 100, 80, 120
        
        # Create simple patterns
        A_vals = [(i + j + 1) % 10 for i in range(m) for j in range(k)]
        B_vals = [(i + j + 1) % 10 for i in range(k) for j in range(n)]
        
        A = create_quad_array(A_vals, shape=(m, k))
        B = create_quad_array(B_vals, shape=(k, n))
        
        result = np.matmul(A, B)
        
        assert result.shape == (m, n)
        
        # Verify that result doesn't contain NaN or inf
        result_flat = result.flatten()
        for i in range(min(10, len(result_flat))):  # Check first few elements
            val = float(result_flat[i])
            assert not np.isnan(val), f"NaN found at position {i}"
            assert not np.isinf(val), f"Inf found at position {i}"


# ================================================================================
# BASIC ERROR HANDLING
# ================================================================================

class TestBasicErrorHandling:
    """Test basic error conditions"""
    
    def test_dimension_mismatch_vectors(self):
        """Test dimension mismatch in vectors"""
        x = create_quad_array([1, 2])
        y = create_quad_array([1, 2, 3])
        
        with pytest.raises(ValueError, match=r"matmul: Input operand 1 has a mismatch in its core dimension 0"):
            np.matmul(x, y)
    
    def test_dimension_mismatch_matrix_vector(self):
        """Test dimension mismatch in matrix-vector"""
        A = create_quad_array([1, 2, 3, 4], shape=(2, 2))
        x = create_quad_array([1, 2, 3])  # Wrong size
        
        with pytest.raises(ValueError, match=r"matmul: Input operand 1 has a mismatch in its core dimension 0"):
            np.matmul(A, x)
    
    def test_dimension_mismatch_matrices(self):
        """Test dimension mismatch in matrix-matrix"""
        A = create_quad_array([1, 2, 3, 4], shape=(2, 2))
        B = create_quad_array([1, 2, 3, 4, 5, 6], shape=(3, 2))  # Wrong size

        with pytest.raises(ValueError, match=r"matmul: Input operand 1 has a mismatch in its core dimension 0"):
            np.matmul(A, B)


def _assert_matmul_matches_float64(A_q, B_q, A_f, B_f, rtol=1e-14, atol=1e-14):
    """Compute matmul on both quad and float64 and check element-wise agreement."""
    got = np.matmul(A_q, B_q)
    ref = np.matmul(A_f, B_f)
    assert got.shape == ref.shape, f"shape mismatch: got {got.shape}, ref {ref.shape}"
    got_flat = got.ravel()
    ref_flat = ref.ravel()
    for i in range(ref_flat.size):
        gv = float(got_flat[i])
        rv = float(ref_flat[i])
        if np.isnan(rv):
            assert np.isnan(gv), f"index {i}: expected NaN, got {gv}"
        elif np.isinf(rv):
            assert np.isinf(gv) and np.sign(gv) == np.sign(rv), \
                f"index {i}: expected inf with sign {np.sign(rv)}, got {gv}"
        else:
            denom = max(abs(rv), 1.0)
            assert abs(gv - rv) <= atol + rtol * denom, \
                f"index {i}: got {gv}, expected {rv}"


class TestMatmulBatched:
    """np.matmul on stacks of matrices (the N-D / broadcast case).

    Every test in this class exercises an outer loop count > 1 (or a code
    path that previously skipped batches), and compares against the
    float64 reference for exact-shape and elementwise correctness.
    """

    # ---------- 3D @ 3D, equal batch dims (the canonical batched-GEMM case) ----------

    @pytest.mark.parametrize("B", [2, 3, 5, 8, 17])
    def test_3d_at_3d_various_batches(self, B):
        rng = np.random.default_rng(seed=B)
        A_f = rng.standard_normal((B, 4, 5))
        B_f = rng.standard_normal((B, 5, 6))
        _assert_matmul_matches_float64(_qnd(A_f), _qnd(B_f), A_f, B_f)

    def test_3d_at_3d_batch_size_one(self):
        """Edge case: leading dim = 1 (still triggers gufunc outer loop)."""
        rng = np.random.default_rng(seed=1)
        A_f = rng.standard_normal((1, 3, 4))
        B_f = rng.standard_normal((1, 4, 5))
        _assert_matmul_matches_float64(_qnd(A_f), _qnd(B_f), A_f, B_f)

    def test_3d_at_3d_distinct_batches_have_distinct_results(self):
        """Verify each batch is actually computed (not a copy of batch 0)."""
        A_f = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
        B_f = np.arange(2 * 4 * 5, dtype=np.float64).reshape(2, 4, 5) + 100.0
        A_q = _qnd(A_f)
        B_q = _qnd(B_f)
        got = np.matmul(A_q, B_q)
        ref = np.matmul(A_f, B_f)
        assert got.shape == ref.shape
        # Each batch differs from the others in the reference
        assert not np.allclose(ref[0], ref[1])
        # And in our result
        for b in range(ref.shape[0]):
            for i in range(ref.shape[1]):
                for j in range(ref.shape[2]):
                    assert float(got[b, i, j]) == ref[b, i, j], \
                        f"batch={b} [{i},{j}]: got {float(got[b,i,j])} vs ref {ref[b,i,j]}"

    # ---------- 3D @ 2D and 2D @ 3D (broadcast one operand) ----------

    def test_3d_at_2d_broadcast(self):
        """A is 3D, B is 2D — B is reused across all batches of A."""
        rng = np.random.default_rng(seed=7)
        A_f = rng.standard_normal((4, 3, 5))
        B_f = rng.standard_normal((5, 6))
        _assert_matmul_matches_float64(_qnd(A_f), _qnd(B_f), A_f, B_f)

    def test_2d_at_3d_broadcast(self):
        """A is 2D, B is 3D — A is reused across all batches of B."""
        rng = np.random.default_rng(seed=8)
        A_f = rng.standard_normal((3, 5))
        B_f = rng.standard_normal((4, 5, 6))
        _assert_matmul_matches_float64(_qnd(A_f), _qnd(B_f), A_f, B_f)

    def test_3d_at_3d_broadcast_size1_right(self):
        """(B,M,K) @ (1,K,N) — the right operand is broadcast over batch."""
        rng = np.random.default_rng(seed=9)
        A_f = rng.standard_normal((4, 3, 5))
        B_f = rng.standard_normal((1, 5, 6))
        _assert_matmul_matches_float64(_qnd(A_f), _qnd(B_f), A_f, B_f)

    def test_3d_at_3d_broadcast_size1_left(self):
        """(1,M,K) @ (B,K,N) — the left operand is broadcast over batch."""
        rng = np.random.default_rng(seed=10)
        A_f = rng.standard_normal((1, 3, 5))
        B_f = rng.standard_normal((4, 5, 6))
        _assert_matmul_matches_float64(_qnd(A_f), _qnd(B_f), A_f, B_f)

    # ---------- 4D / 5D — two or more batch dimensions ----------

    def test_4d_at_4d_two_batch_dims(self):
        rng = np.random.default_rng(seed=11)
        A_f = rng.standard_normal((2, 3, 4, 5))
        B_f = rng.standard_normal((2, 3, 5, 6))
        _assert_matmul_matches_float64(_qnd(A_f), _qnd(B_f), A_f, B_f)

    def test_4d_at_4d_mixed_broadcast(self):
        """(3,1,M,K) @ (1,2,K,N) → (3,2,M,N) — broadcasts on multiple axes."""
        rng = np.random.default_rng(seed=12)
        A_f = rng.standard_normal((3, 1, 4, 5))
        B_f = rng.standard_normal((1, 2, 5, 6))
        _assert_matmul_matches_float64(_qnd(A_f), _qnd(B_f), A_f, B_f)

    def test_5d_at_5d(self):
        rng = np.random.default_rng(seed=13)
        A_f = rng.standard_normal((2, 2, 2, 3, 4))
        B_f = rng.standard_normal((2, 2, 2, 4, 5))
        _assert_matmul_matches_float64(_qnd(A_f), _qnd(B_f), A_f, B_f)

    def test_5d_with_broadcasting(self):
        """High-rank mixed broadcasting."""
        rng = np.random.default_rng(seed=14)
        A_f = rng.standard_normal((2, 1, 3, 4, 5))
        B_f = rng.standard_normal((1, 4, 1, 5, 6))
        _assert_matmul_matches_float64(_qnd(A_f), _qnd(B_f), A_f, B_f)

    # ---------- Batched DOT / GEMV-shaped inputs (different internal paths) ----------

    def test_batched_dot_shaped(self):
        """(B,1,K) @ (B,K,1) → (B,1,1) — exercises the batched DOT branch."""
        rng = np.random.default_rng(seed=15)
        A_f = rng.standard_normal((4, 1, 6))
        B_f = rng.standard_normal((4, 6, 1))
        _assert_matmul_matches_float64(_qnd(A_f), _qnd(B_f), A_f, B_f)

    def test_batched_gemv_shaped(self):
        """(B,M,K) @ (B,K,1) → (B,M,1) — exercises the batched GEMV branch."""
        rng = np.random.default_rng(seed=16)
        A_f = rng.standard_normal((3, 4, 5))
        B_f = rng.standard_normal((3, 5, 1))
        _assert_matmul_matches_float64(_qnd(A_f), _qnd(B_f), A_f, B_f)

    def test_batched_left_singleton_m(self):
        """(B,1,K) @ (B,K,N) → (B,1,N) — m=1 with general right matrix."""
        rng = np.random.default_rng(seed=17)
        A_f = rng.standard_normal((4, 1, 5))
        B_f = rng.standard_normal((4, 5, 6))
        _assert_matmul_matches_float64(_qnd(A_f), _qnd(B_f), A_f, B_f)

    # ---------- 1D operand promotion + batching ----------

    def test_3d_at_1d_collapsed_matvec(self):
        """(B,M,K) @ (K,) → (B,M): 1D right operand promoted then squeezed."""
        rng = np.random.default_rng(seed=18)
        A_f = rng.standard_normal((4, 3, 5))
        x_f = rng.standard_normal((5,))
        _assert_matmul_matches_float64(_qnd(A_f), _qnd(x_f), A_f, x_f)

    def test_1d_at_3d_collapsed_vecmat(self):
        """(K,) @ (B,K,N) → (B,N): 1D left operand promoted then squeezed."""
        rng = np.random.default_rng(seed=19)
        x_f = rng.standard_normal((5,))
        B_f = rng.standard_normal((4, 5, 6))
        _assert_matmul_matches_float64(_qnd(x_f), _qnd(B_f), x_f, B_f)

    # ---------- Non-contiguous / Fortran-order batched matrices ----------

    @pytest.mark.xfail(
        reason="F-contiguous matmul is broken even at 2D — separate pre-existing "
               "bug in the QBLAS dispatch path (treats col-major data as row-major). "
               "Not related to the batch-loop fix; tracked separately."
    )
    def test_batched_fortran_order(self):
        """F-contiguous batched arrays. Currently xfails due to a separate
        pre-existing bug in the matmul aligned strided loop: it always passes
        layout 'R' to qblas_gemm regardless of stride pattern."""
        rng = np.random.default_rng(seed=20)
        A_f = np.asfortranarray(rng.standard_normal((3, 4, 5)))
        B_f = np.asfortranarray(rng.standard_normal((3, 5, 6)))
        A_q = np.asfortranarray(_qnd(A_f))
        B_q = np.asfortranarray(_qnd(B_f))
        _assert_matmul_matches_float64(A_q, B_q, A_f, B_f)

    def test_batched_non_contiguous_slice(self):
        """Slice along the batch dim to produce a non-contiguous view."""
        rng = np.random.default_rng(seed=21)
        # 5 batches, but take every other one
        A_full_f = rng.standard_normal((5, 3, 4))
        B_full_f = rng.standard_normal((5, 4, 6))
        A_f = A_full_f[::2]
        B_f = B_full_f[::2]
        A_q = _qnd(A_full_f)[::2]
        B_q = _qnd(B_full_f)[::2]
        _assert_matmul_matches_float64(A_q, B_q, A_f, B_f)

    @pytest.mark.xfail(
        reason="A.swapaxes view yields a per-batch col-major matrix; same "
               "pre-existing layout-dispatch bug as test_batched_fortran_order. "
               "Not related to the batch-loop fix; tracked separately."
    )
    def test_batched_transposed_view(self):
        """A.swapaxes-based view of a 3D array as the batched matrix.

        After swapping the last two axes, each batch is col-major. Like
        F-order arrays, this hits the same pre-existing layout-dispatch bug.
        """
        rng = np.random.default_rng(seed=22)
        A_full_f = rng.standard_normal((3, 5, 4))  # shape (B, K, M)
        A_f = A_full_f.swapaxes(-2, -1)            # → (B, M=4, K=5) via stride trick
        B_f = rng.standard_normal((3, 5, 6))       # (B, K=5, N=6)
        A_q = _qnd(A_full_f).swapaxes(-2, -1)
        B_q = _qnd(B_f)
        assert not A_q.flags.c_contiguous
        _assert_matmul_matches_float64(A_q, B_q, A_f, B_f)

    # ---------- Special-value propagation across batches ----------

    def test_nan_in_one_batch_only(self):
        """A NaN in batch[1] must not contaminate batch[0]'s result."""
        A_f = np.ones((2, 2, 3), dtype=np.float64)
        B_f = np.ones((2, 3, 2), dtype=np.float64)
        A_f[1, 0, 0] = float('nan')
        A_q = _qnd(A_f)
        B_q = _qnd(B_f)
        got = np.matmul(A_q, B_q)
        ref = np.matmul(A_f, B_f)
        # Batch 0 should be untouched and finite
        for i in range(2):
            for j in range(2):
                assert float(got[0, i, j]) == ref[0, i, j]
        # Batch 1 row 0 should contain NaNs (NaN propagation)
        assert np.isnan(float(got[1, 0, 0]))
        assert np.isnan(float(got[1, 0, 1]))

    def test_inf_in_one_batch_only(self):
        """An infinity in batch[1] must not affect batch[0]."""
        A_f = np.ones((2, 2, 3), dtype=np.float64)
        B_f = np.ones((2, 3, 2), dtype=np.float64)
        A_f[1, 1, 1] = float('inf')
        A_q = _qnd(A_f)
        B_q = _qnd(B_f)
        got = np.matmul(A_q, B_q)
        ref = np.matmul(A_f, B_f)
        for i in range(2):
            for j in range(2):
                assert float(got[0, i, j]) == ref[0, i, j]
        assert np.isinf(float(got[1, 1, 0]))
        assert np.isinf(float(got[1, 1, 1]))

    # ---------- out= kwarg with batched inputs ----------

    def test_out_kwarg_with_batched(self):
        """User-provided `out=` must be filled across all batches."""
        rng = np.random.default_rng(seed=23)
        A_f = rng.standard_normal((3, 4, 5))
        B_f = rng.standard_normal((3, 5, 6))
        A_q = _qnd(A_f)
        B_q = _qnd(B_f)
        out_q = np.zeros((3, 4, 6), dtype=QuadPrecDType(backend='sleef'))
        result = np.matmul(A_q, B_q, out=out_q)
        ref = np.matmul(A_f, B_f)
        assert result is out_q
        for b in range(3):
            for i in range(4):
                for j in range(6):
                    assert abs(float(out_q[b, i, j]) - ref[b, i, j]) < 1e-12

    # ---------- Larger batched sizes ----------

    @pytest.mark.parametrize("batch,m,k,n", [
        (10, 3, 4, 5),
        (4, 8, 8, 8),
        (32, 2, 2, 2),
    ])
    def test_batched_larger(self, batch, m, k, n):
        rng = np.random.default_rng(seed=42 + batch + m + k + n)
        A_f = rng.standard_normal((batch, m, k))
        B_f = rng.standard_normal((batch, k, n))
        _assert_matmul_matches_float64(_qnd(A_f), _qnd(B_f), A_f, B_f,
                                       rtol=1e-13, atol=1e-13)