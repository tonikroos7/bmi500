import math
import numbers
import random
import time

# Container types that have len() and indexing but are never valid vectors
_REJECTED_CONTAINERS = (str, bytes, bytearray, dict, set, frozenset)


def _check_sequence(obj, name, expected_ndim):
    """
    Check that obj is an indexable, sized container with the expected
    number of dimensions (1 for a vector, 2 for a matrix).

    Raises:
        TypeError: If obj is not a supported sequence type.
        ValueError: If obj is an array with the wrong number of dimensions.
    """
    # Strings, bytes, dicts and sets have len() but are not numeric vectors
    # (a dict with keys 0..n-1 would otherwise be silently accepted)
    if isinstance(obj, _REJECTED_CONTAINERS):
        raise TypeError(
            f"{name} must be a list, tuple, or NumPy array, "
            f"got {type(obj).__name__}"
        )

    # Generators and other iterators have no len() and cannot be indexed
    if not (hasattr(obj, "__len__") and hasattr(obj, "__getitem__")):
        hint = " (wrap it in list() first)" if hasattr(obj, "__iter__") else ""
        raise TypeError(
            f"{name} must be an indexable sequence, "
            f"got {type(obj).__name__}{hint}"
        )

    # NumPy arrays report their dimensionality; reject e.g. an (n, 1)
    # column vector where a 1-D vector is expected
    ndim = getattr(obj, "ndim", None)
    if ndim is not None and ndim != expected_ndim:
        shape = getattr(obj, "shape", "?")
        hint = " (use .ravel() to flatten)" if expected_ndim == 1 else ""
        raise ValueError(
            f"{name} must be {expected_ndim}-D, got {ndim}-D array "
            f"with shape {shape}{hint}"
        )


def _check_scalar(x, name, index):
    """
    Check that x is a real number (bool and complex are rejected).

    Called only for elements whose type is not exactly float or int, so the
    common case stays fast.

    Raises:
        TypeError: If x is not a real number.
    """
    # bool is a subclass of int, so it must be excluded explicitly;
    # numbers.Real also rejects complex, Decimal, None, str, and lists
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        raise TypeError(
            f"{name}[{index}] must be a real number, got {type(x).__name__}"
        )


def _find_nonfinite(v, name):
    """
    Raise ValueError naming the first NaN/inf element of v, if any.

    Only called after a non-finite result, so it costs nothing on the
    normal path.
    """
    for i in range(len(v)):
        try:
            finite = math.isfinite(v[i])
        except OverflowError:
            # A huge int is finite; it just cannot be converted to float
            continue
        if not finite:
            raise ValueError(
                f"{name}[{i}] is {v[i]}; pass check_finite=False to allow NaN/inf"
            )


# create a function to compute the dot product of two vectors using a for loop
# add comments for the selected function
def dot_product(a, b, *, check_finite=True):
    """
    Compute the dot product of two real vectors using an explicit for loop.

    Args:
        a (sequence of real): First vector of length n (list, tuple, or 1-D array).
        b (sequence of real): Second vector of length n.
        check_finite (bool): If True (default), reject NaN/inf inputs and
            raise if the result overflows. Set to False to let NaN/inf
            propagate, e.g. for masked image data.

    Returns:
        float: a[0]*b[0] + a[1]*b[1] + ... + a[n-1]*b[n-1]

    Raises:
        TypeError: If an input is not a valid vector or holds a non-real value.
        ValueError: If lengths differ, an input is not 1-D, or (with
            check_finite) an element is NaN/inf.
        OverflowError: If an element is too large for a float, or (with
            check_finite) the result overflows.
    """
    # Both inputs must be 1-D indexable sequences
    _check_sequence(a, "a", expected_ndim=1)
    _check_sequence(b, "b", expected_ndim=1)

    # The dot product is only defined for vectors of equal length
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")

    # Running sum of element-wise products (start at 0.0 so the result is a float)
    result = 0.0

    # Validate each pair of elements, then add their product to the total.
    # Plain float/int values take a fast path; everything else (NumPy
    # scalars, Fraction, and invalid values) goes through _check_scalar.
    for i in range(len(a)):
        x = a[i]
        y = b[i]
        if type(x) is not float and type(x) is not int:
            _check_scalar(x, "a", i)
        if type(y) is not float and type(y) is not int:
            _check_scalar(y, "b", i)
        try:
            result += x * y
        except OverflowError:
            # Raised when a Python int is too large to convert to float
            raise OverflowError(
                f"a[{i}] * b[{i}] is too large to convert to float"
            ) from None

    # NaN/inf never cancel back to a finite value, so a non-finite result
    # means either a non-finite input or overflow. Checking once here
    # avoids an isfinite() call per element.
    if check_finite and not math.isfinite(result):
        _find_nonfinite(a, "a")
        _find_nonfinite(b, "b")
        raise OverflowError(
            "Dot product overflowed to a non-finite value; "
            "consider rescaling the inputs"
        )

    return float(result)


# create a function to compute the matrix-vector product using the dot_product function
# add comments for the selected function
def matrix_vector_product(matrix, vector, *, check_finite=True):
    """
    Compute y = A @ x for an m x n real matrix A and a length-n real vector x.

    Each entry of the output is the dot product of one row of the matrix
    with the input vector: y[i] = dot_product(A[i], x).

    All structural checks (types, dimensions, row lengths) run before any
    arithmetic, so a bad row near the end of a large matrix fails fast.
    Element checks run inside dot_product.

    Args:
        matrix (sequence of sequences of real): m x n matrix as a list of
            m rows, or a 2-D NumPy array.
        vector (sequence of real): Vector of length n (list, tuple, or 1-D array).
        check_finite (bool): Passed through to dot_product.

    Returns:
        list[float]: Result vector of length m.

    Raises:
        TypeError: If the matrix, a row, or the vector has the wrong type.
        ValueError: If dimensions do not match or rows have unequal lengths.
        OverflowError: See dot_product.
    """
    # The matrix must be 2-D and the vector 1-D
    _check_sequence(matrix, "matrix", expected_ndim=2)
    _check_sequence(vector, "vector", expected_ndim=1)

    n_rows = len(matrix)

    # Use len() rather than `if not matrix`, which fails on NumPy arrays
    if n_rows == 0:
        # An empty NumPy array still records its column count, so check it
        shape = getattr(matrix, "shape", None)
        if shape is not None and shape[1] != len(vector):
            raise ValueError(
                f"Dimension mismatch: matrix has shape {shape}, "
                f"vector has length {len(vector)}"
            )
        return []

    # Every row must be a 1-D sequence; this also catches a flat list
    # passed as the matrix and swapped arguments
    first = matrix[0]
    if isinstance(first, numbers.Number):
        raise TypeError(
            "matrix must be 2-D (a sequence of rows), but matrix[0] is a "
            "number; check that the arguments are not swapped"
        )
    _check_sequence(first, "matrix[0]", expected_ndim=1)
    n_cols = len(first)

    # Vector length must equal the number of columns
    if len(vector) != n_cols:
        raise ValueError(
            f"Dimension mismatch: matrix has {n_cols} columns, "
            f"vector has length {len(vector)}"
        )

    # Check every row's type and length before computing anything, so a
    # ragged row is reported by index instead of failing partway through
    for i in range(1, n_rows):
        row = matrix[i]
        _check_sequence(row, f"matrix[{i}]", expected_ndim=1)
        if len(row) != n_cols:
            raise ValueError(
                f"Ragged matrix: row {i} has length {len(row)}, "
                f"expected {n_cols}"
            )

    # Compute one output entry per row by dotting that row with the vector
    result = []
    for i in range(n_rows):
        try:
            result.append(dot_product(matrix[i], vector, check_finite=check_finite))
        except (TypeError, ValueError, OverflowError) as err:
            # Add the row index so errors in large matrices are easy to locate
            raise type(err)(
                f"In row {i} (a = matrix row, b = vector): {err}"
            ) from err

    return result


# create a main function to test the matrix-vector product function using randomly generated data of size 1000x1000
# add comments for the selected function
def main():
    """
    Test matrix_vector_product on a random 1000x1000 matrix and a random
    length-1000 vector, then check that the input guards reject bad inputs.

    Steps:
        1. Generate reproducible random data.
        2. Time the matrix-vector product.
        3. Check the output length.
        4. Spot-check several rows against an independent computation.
        5. If NumPy is installed, compare the full result against NumPy.
        6. Confirm that a few representative bad inputs raise errors.
    """
    size = 1000

    # Fix the seed so every run uses the same data
    random.seed(42)

    # Build a size x size matrix and a length-size vector of floats in [0, 1)
    matrix = [[random.random() for _ in range(size)] for _ in range(size)]
    vector = [random.random() for _ in range(size)]

    # Time the pure-Python matrix-vector product
    start = time.perf_counter()
    result = matrix_vector_product(matrix, vector)
    elapsed = time.perf_counter() - start

    # The output must have one entry per matrix row
    assert len(result) == size, f"Expected length {size}, got {len(result)}"

    # Spot-check 5 random rows with an independent sum/zip computation
    for i in random.sample(range(size), 5):
        expected = sum(m * v for m, v in zip(matrix[i], vector))
        assert math.isclose(result[i], expected, rel_tol=1e-12), (
            f"Row {i}: got {result[i]}, expected {expected}"
        )

    print(f"Matrix size:       {size}x{size}")
    print(f"Elapsed time:      {elapsed:.3f} s")
    print(f"First 5 entries:   {[round(x, 4) for x in result[:5]]}")
    print("Spot checks:       passed")

    # Optional full comparison against NumPy, skipped if it is not installed
    try:
        import numpy as np

        np_matrix = np.asarray(matrix)
        np_vector = np.asarray(vector)
        np_result = np_matrix @ np_vector
        max_err = float(np.max(np.abs(np.asarray(result) - np_result)))
        assert max_err < 1e-9, f"NumPy mismatch, max abs error = {max_err}"
        print(f"NumPy check:       passed (max abs error = {max_err:.2e})")

        # NumPy arrays can also be passed in directly
        direct = matrix_vector_product(np_matrix[:10], np_vector)
        assert np.allclose(direct, np_result[:10])
        print("NumPy input:       passed")
    except ImportError:
        print("NumPy check:       skipped (NumPy not installed)")

    # Each bad input must raise the expected exception type
    bad_inputs = [
        ("ragged row", lambda: matrix_vector_product([[1, 2], [3]], [1, 1]), ValueError),
        ("swapped args", lambda: matrix_vector_product([1, 2], [[1, 2]]), TypeError),
        ("NaN element", lambda: dot_product([float("nan")], [1.0]), ValueError),
        ("overflow", lambda: dot_product([1e200], [1e200]), OverflowError),
        ("bool element", lambda: dot_product([True], [1]), TypeError),
        ("generator", lambda: dot_product((x for x in [1]), [1]), TypeError),
    ]
    for label, call, expected_error in bad_inputs:
        try:
            call()
        except expected_error:
            continue
        raise AssertionError(f"Guard failed: {label} did not raise {expected_error.__name__}")
    print(f"Guard checks:      passed ({len(bad_inputs)} cases)")


if __name__ == "__main__":
    main()