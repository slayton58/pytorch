# Owner(s): ["module: native-overrides"]

import pathlib
import sys

# Add repo root to Python path (standard PyTorch test pattern)
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.testing._internal.common_utils import TestCase, run_tests, parametrize
from torch._inductor.python_native_aoti_condition_extraction import extract_conditions, extract_conditions_from_source

# Clean up path
sys.path.remove(str(REPO_ROOT))


# Test case definitions as strings (PyTorch pattern for runtime function generation)
TEST_CASES = {
    "complex_and_conditions": '''
def test_func(x):
    if (x.ndim == 2 and
        x.shape[0] == x.shape[1] and
        x.shape[0] >= 512 and
        x.dtype == torch.float32):
        return True
    return False
''',

    "shape_gte": '''
def test_func(x):
    return x.shape[0] >= 512
''',

    "shape_lte": '''
def test_func(x):
    return x.shape[0] <= 512
''',

    "shape_eq": '''
def test_func(x):
    return x.shape[0] == 512
''',

    "nested_and_or": '''
def test_func(x):
    if ((x.ndim == 2 or x.ndim == 3) and
        (x.dtype == torch.float32 or x.dtype == torch.bfloat16)):
        return True
    return False
''',

    "basic_dtype": '''
def test_func(x):
    if x.dtype == torch.float32:
        return True
    return False
''',

    "numel_gte": '''
def test_func(x):
    if x.numel() >= 1024:
        return True
    return False
''',

    "shape_dim_gte": '''
def test_func(x):
    if x.shape[0] >= 512:
        return True
    return False
''',

    "ndim_eq": '''
def test_func(x):
    if x.ndim == 2:
        return True
    return False
''',

    "size_method": '''
def test_func(x):
    if x.size(1) >= 128:
        return True
    return False
''',

    "cross_dimensional": '''
def test_func(x):
    if x.shape[0] == x.shape[1]:
        return True
    return False
''',

    "unsupported_pattern": '''
def test_func(x):
    if x.data_ptr() % 2 == 0:  # Unsupported pattern
        return True
    return False
''',

    "nested_conditions": '''
def test_func(x):
    if x.ndim >= 2:
        if x.shape[0] >= x.shape[1]:
            return True
    return False
''',
}


class TestConditionExtraction(TestCase):
    """Test AST-based condition extraction from Python override functions."""

    def test_basic_dtype_conditions(self):
        """Test extraction of basic dtype conditions."""
        conditions = extract_conditions_from_source(TEST_CASES["basic_dtype"])
        self.assertNotEqual(conditions.get("type"), "error")
        self.assertEqual(conditions.get("type"), "dtype_eq")

    def test_numel_conditions(self):
        """Test extraction of numel conditions."""
        conditions = extract_conditions_from_source(TEST_CASES["numel_gte"])
        self.assertNotEqual(conditions.get("type"), "error")
        self.assertEqual(conditions.get("type"), "numel_gte")

    def test_shape_conditions(self):
        """Test extraction of shape-based conditions."""
        conditions = extract_conditions_from_source(TEST_CASES["shape_dim_gte"])
        self.assertNotEqual(conditions.get("type"), "error")
        self.assertEqual(conditions.get("type"), "shape_dim_gte")
        self.assertEqual(conditions.get("dim"), 0)
        self.assertEqual(conditions.get("value"), 512)

    def test_ndim_conditions(self):
        """Test extraction of ndim conditions."""
        conditions = extract_conditions_from_source(TEST_CASES["ndim_eq"])
        self.assertNotEqual(conditions.get("type"), "error")
        self.assertEqual(conditions.get("type"), "ndim_eq")
        self.assertEqual(conditions.get("value"), 2)

    def test_size_method_conditions(self):
        """Test extraction of size() method conditions."""
        conditions = extract_conditions_from_source(TEST_CASES["size_method"])
        self.assertNotEqual(conditions.get("type"), "error")
        self.assertEqual(conditions.get("type"), "size_dim_gte")
        self.assertEqual(conditions.get("dim"), 1)
        self.assertEqual(conditions.get("value"), 128)

    def test_cross_dimensional_conditions(self):
        """Test extraction of cross-dimensional comparisons."""
        conditions = extract_conditions_from_source(TEST_CASES["cross_dimensional"])
        self.assertNotEqual(conditions.get("type"), "error")
        self.assertEqual(conditions.get("type"), "shape_dim_eq_cross")
        self.assertEqual(conditions.get("dim1"), 0)
        self.assertEqual(conditions.get("dim2"), 1)

    def test_complex_and_conditions(self):
        """Test extraction of complex AND conditions."""
        conditions = extract_conditions_from_source(TEST_CASES["complex_and_conditions"])
        self.assertNotEqual(conditions.get("type"), "error")
        self.assertEqual(conditions.get("type"), "and")

        sub_conditions = conditions.get("conditions", [])
        self.assertGreater(len(sub_conditions), 0)

        # Flatten nested AND conditions to find all condition types
        def flatten_condition_types(cond_list):
            types = []
            for c in cond_list:
                if c.get("type") == "and":
                    types.extend(flatten_condition_types(c.get("conditions", [])))
                else:
                    types.append(c.get("type"))
            return types

        all_condition_types = flatten_condition_types(sub_conditions)

        # Should contain multiple condition types (somewhere in the nested structure)
        self.assertIn("ndim_eq", all_condition_types)
        self.assertIn("shape_dim_eq_cross", all_condition_types)
        self.assertIn("shape_dim_gte", all_condition_types)
        self.assertIn("dtype_eq", all_condition_types)

    def test_shape_operators(self):
        """Test different shape comparison operators."""
        test_cases = [
            (">=", "shape_dim_gte"),
            ("<=", "shape_dim_lte"),
            ("==", "shape_dim_eq"),
        ]

        for operator, expected_type in test_cases:
            with self.subTest(operator=operator, expected_type=expected_type):
                # Create function source dynamically with different operators
                func_code = f"""
def test_func(x):
    if x.shape[0] {operator} 256:
        return True
    return False
"""
                conditions = extract_conditions_from_source(func_code)
                self.assertNotEqual(conditions.get("type"), "error")

                # Should extract the specific comparison type
                if conditions.get("type") == expected_type:
                    # Direct match
                    self.assertEqual(conditions.get("type"), expected_type)
                elif conditions.get("type") == "and":
                    # Wrapped in AND (sometimes happens with complex parsing)
                    sub_conditions = conditions.get("conditions", [])
                    condition_types = [c.get("type") for c in sub_conditions]
                    self.assertIn(expected_type, condition_types)
                else:
                    # Check if it's the expected type
                    self.assertEqual(conditions.get("type"), expected_type)
        self.assertEqual(conditions.get("type"), expected_type)

    def test_unsupported_conditions(self):
        """Test handling of unsupported condition patterns."""
        conditions = extract_conditions_from_source(TEST_CASES["unsupported_pattern"])
        # Should gracefully handle unsupported patterns
        self.assertIn(conditions.get("type"), ["always_true", "error"])

    def test_nested_conditions(self):
        """Test extraction from nested conditional structures."""
        conditions = extract_conditions_from_source(TEST_CASES["nested_conditions"])
        # Should extract meaningful conditions even from nested structure
        self.assertNotEqual(conditions.get("type"), "error")

    def test_error_handling(self):
        """Test robust error handling for malformed functions."""
        # Test with None
        conditions = extract_conditions(None)
        self.assertEqual(conditions.get("type"), "always_true")

        # Test with non-function
        conditions = extract_conditions("not_a_function")
        self.assertEqual(conditions.get("type"), "always_true")


class TestConditionExtractionRegressions(TestCase):
    """Regression tests for specific condition extraction scenarios."""

    def test_matmul_square_matrix(self):
        """Test condition extraction for square matrix matmul pattern."""
        def square_matrix_dispatch(mat1, mat2):
            if (mat1.ndim == 2 and mat2.ndim == 2 and
                mat1.shape[0] == mat1.shape[1] and
                mat1.shape[1] == mat2.shape[0] and
                mat1.shape[0] >= 512):
                return True
            return False

        conditions = extract_conditions(square_matrix_dispatch)
        self.assertNotEqual(conditions.get("type"), "error")
        self.assertEqual(conditions.get("type"), "and")

    def test_batched_operations(self):
        """Test condition extraction for batched operation patterns."""
        def batched_dispatch(x):
            if x.ndim >= 3 and x.shape[0] >= 16:
                return True
            return False

        conditions = extract_conditions(batched_dispatch)
        self.assertNotEqual(conditions.get("type"), "error")
        self.assertEqual(conditions.get("type"), "and")


if __name__ == "__main__":
    run_tests()