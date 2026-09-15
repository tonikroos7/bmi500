import copy
import random

import pytest

from matvec_multiply import dot_product, matrix_vector_product


# generate tests for matrix-vector-product function in matvec_multiply.py
class TestMatrixVectorProduct:
    """Tests for matrix_vector_product(matrix, vector)."""

    def test_known_square_example(self):
        # Hand-computed 2x2 example: [[1, 2], [3, 4]] @ [5, 6] = [17, 39]
        matrix = [[1, 2], [3, 4]]
        vector = [5, 6]
        assert matrix_vector_product(matrix, vector) == [17, 39]

    def test_known_non_square_example(self):
        # 2x3 matrix times length-3 vector gives a length-2 result
        matrix = [[1, 0, 2], [-1, 3, 1]]
        vector = [3, 2, 1]
        assert matrix_vector_product(matrix, vector) == [5, 4]

    def test_tall_matrix_output_length(self):
        # 4x2 matrix: output length equals the number of rows, not columns
        matrix = [[1, 1], [2, 2], [3, 3], [4, 4]]
        vector = [1, -1]
        result = matrix_vector_product(matrix, vector)
        assert len(result) == 4
        assert result == [0, 0, 0, 0]

    def test_one_by_one(self):
        # Smallest non-empty case reduces to scalar multiplication
        assert matrix_vector_product([[7.0]], [3.0]) == [21.0]

    def test_identity_matrix_returns_input(self):
        # I @ x == x
        n = 5
        identity = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        vector = [1.5, -2.0, 0.0, 3.25, 10.0]
        assert matrix_vector_product(identity, vector) == vector

    def test_zero_matrix_returns_zeros(self):
        # 0 @ x == 0
        matrix = [[0.0] * 3 for _ in range(4)]
        assert matrix_vector_product(matrix, [1.0, 2.0, 3.0]) == [0.0] * 4

    def test_zero_vector_returns_zeros(self):
        # A @ 0 == 0
        matrix = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        assert matrix_vector_product(matrix, [0.0, 0.0]) == [0.0, 0.0, 0.0]

    def test_diagonal_matrix_scales_entries(self):
        # A diagonal matrix scales each entry of x by the diagonal value
        diag = [2.0, -1.0, 0.5]
        matrix = [[diag[i] if i == j else 0.0 for j in range(3)] for i in range(3)]
        vector = [4.0, 4.0, 4.0]
        assert matrix_vector_product(matrix, vector) == [8.0, -4.0, 2.0]

    def test_empty_matrix_returns_empty_list(self):
        # No rows means no output entries
        assert matrix_vector_product([], []) == []

    def test_dimension_mismatch_raises(self):
        # Vector length must equal the number of columns
        matrix = [[1, 2, 3], [4, 5, 6]]
        with pytest.raises(ValueError):
            matrix_vector_product(matrix, [1, 2])

    def test_ragged_matrix_raises(self):
        # A later row with the wrong length is caught by dot_product
        matrix = [[1, 2], [3, 4, 5]]
        with pytest.raises(ValueError):
            matrix_vector_product(matrix, [1, 1])

    def test_inputs_not_modified(self):
        # The function must not mutate its arguments
        matrix = [[1.0, 2.0], [3.0, 4.0]]
        vector = [5.0, 6.0]
        matrix_before = copy.deepcopy(matrix)
        vector_before = list(vector)
        matrix_vector_product(matrix, vector)
        assert matrix == matrix_before
        assert vector == vector_before

    def test_linearity(self):
        # A @ (a*x + b*y) == a*(A @ x) + b*(A @ y)
        rng = random.Random(0)
        n = 20
        matrix = [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
        x = [rng.uniform(-1, 1) for _ in range(n)]
        y = [rng.uniform(-1, 1) for _ in range(n)]
        a, b = 2.5, -0.75

        combined = [a * xi + b * yi for xi, yi in zip(x, y)]
        lhs = matrix_vector_product(matrix, combined)

        ax = matrix_vector_product(matrix, x)
        ay = matrix_vector_product(matrix, y)
        rhs = [a * p + b * q for p, q in zip(ax, ay)]

        assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)

    @pytest.mark.parametrize("rows, cols", [(1, 1), (3, 7), (50, 50), (1000, 1000)])
    def test_matches_numpy(self, rows, cols):
        # Compare against NumPy on random data of several shapes,
        # including the 1000x1000 case used in main()
        np = pytest.importorskip("numpy")
        rng = random.Random(rows * 10_000 + cols)
        matrix = [[rng.random() for _ in range(cols)] for _ in range(rows)]
        vector = [rng.random() for _ in range(cols)]

        result = matrix_vector_product(matrix, vector)
        expected = np.asarray(matrix) @ np.asarray(vector)

        assert len(result) == rows
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


# generate tests for dot-product function in matvec_multiply.py
class TestDotProduct:
    """Tests for dot_product(a, b)."""

    def test_known_integer_example(self):
        # 1*4 + 2*5 + 3*6 = 32
        assert dot_product([1, 2, 3], [4, 5, 6]) == 32

    def test_known_float_example(self):
        # 0.1*0.4 + 0.2*0.5 + 0.3*0.6 = 0.32 (approx because of floating point)
        assert dot_product([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]) == pytest.approx(0.32)

    def test_negative_values(self):
        # (-1)*2 + 3*(-4) + (-5)*(-6) = -2 - 12 + 30 = 16
        assert dot_product([-1, 3, -5], [2, -4, -6]) == 16

    def test_single_element(self):
        # One-element vectors reduce to scalar multiplication
        assert dot_product([3.0], [-2.5]) == -7.5

    def test_empty_vectors_return_zero(self):
        # The sum over no elements is 0.0
        result = dot_product([], [])
        assert result == 0.0
        assert isinstance(result, float)

    def test_returns_float_for_integer_input(self):
        # The accumulator starts at 0.0, so the result is always a float
        assert isinstance(dot_product([1, 2], [3, 4]), float)

    def test_zero_vector(self):
        # Dotting with the zero vector gives 0
        assert dot_product([0, 0, 0], [7, -8, 9]) == 0.0

    def test_orthogonal_vectors(self):
        # Perpendicular vectors have a dot product of 0
        assert dot_product([1, 0, 0], [0, 1, 0]) == 0.0
        assert dot_product([1, 1], [1, -1]) == 0.0

    def test_self_dot_is_squared_norm(self):
        # x . x == ||x||^2, e.g. [3, 4] . [3, 4] = 25
        assert dot_product([3, 4], [3, 4]) == 25

    def test_commutative(self):
        # a . b == b . a
        rng = random.Random(1)
        a = [rng.uniform(-10, 10) for _ in range(100)]
        b = [rng.uniform(-10, 10) for _ in range(100)]
        assert dot_product(a, b) == pytest.approx(dot_product(b, a), rel=1e-12)

    def test_scalar_multiplication(self):
        # (k*a) . b == k * (a . b)
        rng = random.Random(2)
        a = [rng.uniform(-1, 1) for _ in range(50)]
        b = [rng.uniform(-1, 1) for _ in range(50)]
        k = 3.7
        scaled = [k * x for x in a]
        assert dot_product(scaled, b) == pytest.approx(k * dot_product(a, b), rel=1e-12)

    def test_distributive(self):
        # a . (b + c) == a . b + a . c
        rng = random.Random(3)
        a = [rng.uniform(-1, 1) for _ in range(50)]
        b = [rng.uniform(-1, 1) for _ in range(50)]
        c = [rng.uniform(-1, 1) for _ in range(50)]
        b_plus_c = [x + y for x, y in zip(b, c)]
        assert dot_product(a, b_plus_c) == pytest.approx(
            dot_product(a, b) + dot_product(a, c), rel=1e-12, abs=1e-12
        )

    def test_works_with_tuples(self):
        # Any indexable sequence with len() should work, not just lists
        assert dot_product((1, 2), (3, 4)) == 11

    def test_inputs_not_modified(self):
        # The function must not mutate its arguments
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        dot_product(a, b)
        assert a == [1.0, 2.0, 3.0]
        assert b == [4.0, 5.0, 6.0]

    @pytest.mark.parametrize("a, b", [([1, 2, 3], [1, 2]), ([], [1]), ([1], [])])
    def test_length_mismatch_raises(self, a, b):
        # Vectors of different lengths are rejected
        with pytest.raises(ValueError):
            dot_product(a, b)

    @pytest.mark.parametrize("n", [1, 10, 1000])
    def test_matches_numpy(self, n):
        # Compare against NumPy on random vectors of several lengths
        np = pytest.importorskip("numpy")
        rng = random.Random(n)
        a = [rng.uniform(-1, 1) for _ in range(n)]
        b = [rng.uniform(-1, 1) for _ in range(n)]
        assert dot_product(a, b) == pytest.approx(float(np.dot(a, b)), rel=1e-12, abs=1e-12)