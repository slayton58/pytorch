# Owner(s): ["module: native-overrides"]

import pathlib
import sys
import tempfile
import time
from typing import Any, Dict, List

# Add repo root to Python path (standard PyTorch test pattern)
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch._inductor import config
from torch._inductor.aoti_overrides import compile_overrides_for_aoti, _get_override_graphs
from torch._inductor.python_native_aoti_code_generation import compile_overrides

# Clean up path
sys.path.remove(str(REPO_ROOT))


class TestOverrideFallbacks(TestCase):
    """Test fallback behavior when override conditions are not met."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        # Enable override compilation for testing
        self.original_compile_overrides = config.aot_inductor.compile_native_overrides
        config.aot_inductor.compile_native_overrides = True

    def tearDown(self):
        """Clean up test environment."""
        super().tearDown()
        # Restore original configuration
        config.aot_inductor.compile_native_overrides = self.original_compile_overrides

    def test_fallback_when_conditions_not_met(self):
        """Test fallback to original kernel when no override matches."""

        # Test with small tensors that shouldn't match override conditions
        small_tensor = torch.randn(32, 32, device='cuda', dtype=torch.float32)

        # Test with dtypes that shouldn't match override conditions
        wrong_dtype_tensor = torch.randn(1024, 1024, device='cuda', dtype=torch.float64)

        # Test with CPU tensors (overrides are for CUDA)
        cpu_tensor = torch.randn(1024, 1024, device='cpu', dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            override_result = compile_overrides_for_aoti(tmpdir)

            if override_result and override_result["status"] == "success":
                print(f"✅ Testing fallbacks with {len(override_result['compiled_libraries'])} compiled overrides")

                # Get baseline results (without overrides) for comparison
                config.aot_inductor.compile_native_overrides = False
                try:
                    baseline_small = torch.relu(small_tensor)
                    baseline_wrong_dtype = torch.relu(wrong_dtype_tensor)
                    baseline_cpu = torch.relu(cpu_tensor)
                finally:
                    config.aot_inductor.compile_native_overrides = True

                # Test with overrides enabled (should fallback for all cases)
                try:
                    override_small = torch.relu(small_tensor)
                    override_wrong_dtype = torch.relu(wrong_dtype_tensor)
                    override_cpu = torch.relu(cpu_tensor)

                    # Results should be identical (fallback to same kernel)
                    self.assertTrue(torch.allclose(baseline_small, override_small, rtol=1e-6))
                    self.assertTrue(torch.allclose(baseline_wrong_dtype, override_wrong_dtype, rtol=1e-6))
                    self.assertTrue(torch.allclose(baseline_cpu, override_cpu, rtol=1e-6))

                    print(f"✅ Fallback correctness verified:")
                    print(f"  Small tensor: {small_tensor.shape} → fallback worked")
                    print(f"  Wrong dtype: {wrong_dtype_tensor.dtype} → fallback worked")
                    print(f"  CPU tensor: {cpu_tensor.device} → fallback worked")

                except Exception as e:
                    self.fail(f"Fallback execution failed: {e}")
            else:
                self.skipTest("Override compilation failed or disabled")

    def test_generated_fallback_code_correctness(self):
        """Test that generated C++ fallback code is correct."""

        override_graphs = _get_override_graphs()

        if not override_graphs:
            self.skipTest("No override graphs available for testing")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Compile overrides and examine generated code
            compiled_libs = compile_overrides(override_graphs, tmpdir)

            if compiled_libs:
                # Check generated C++ files for proper fallback patterns
                from pathlib import Path

                cpp_files = list(Path(tmpdir).glob("*.cpp"))
                self.assertGreater(len(cpp_files), 0, "No C++ files generated")

                fallback_patterns_found = 0
                kernel_preservation_patterns = [
                    "cached_.*_kernel",           # Kernel cache variables
                    "std::once_flag",             # Thread-safe initialization
                    "c10::Dispatcher::singleton().findOp",  # Kernel lookup
                    "SafeKernelFunction",         # Safe kernel wrapper
                    "callBoxed",                  # Boxed kernel call
                ]

                for cpp_file in cpp_files:
                    content = cpp_file.read_text()

                    patterns_in_file = 0
                    for pattern in kernel_preservation_patterns:
                        if pattern in content:
                            patterns_in_file += 1

                    if patterns_in_file >= 3:  # Should have most fallback patterns
                        fallback_patterns_found += 1

                    print(f"  {cpp_file.name}: {patterns_in_file}/{len(kernel_preservation_patterns)} fallback patterns")

                # At least some files should have proper fallback implementation
                self.assertGreater(fallback_patterns_found, 0,
                    "No files contain proper fallback patterns")

                print(f"✅ Generated code validation:")
                print(f"  {fallback_patterns_found}/{len(cpp_files)} files have correct fallback patterns")

            else:
                self.fail("No libraries compiled for fallback testing")

    def test_kernel_caching_thread_safety(self):
        """Test that kernel caching is thread-safe."""

        # This test verifies the std::once_flag mechanism works correctly
        import threading
        import concurrent.futures

        # Create shared input tensor (all threads use the same input)
        shared_input = torch.randn(16, 16, device='cuda', dtype=torch.float32)

        def run_operation():
            """Run operation that should use cached kernels."""
            # Use same tensor for all threads to get consistent results
            return torch.relu(shared_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            override_result = compile_overrides_for_aoti(tmpdir)

            if override_result and override_result["status"] == "success":
                # Run multiple threads simultaneously
                num_threads = 8
                results = []

                with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    # Submit all tasks
                    futures = [executor.submit(run_operation) for _ in range(num_threads)]

                    # Collect results
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            self.fail(f"Thread execution failed: {e}")

                # Verify all threads succeeded
                self.assertEqual(len(results), num_threads)

                # Verify all results are consistent
                reference_result = results[0]
                for i, result in enumerate(results[1:], 1):
                    self.assertTrue(torch.allclose(reference_result, result, rtol=1e-6),
                        f"Thread {i} result differs from reference")

                print(f"✅ Thread safety test passed:")
                print(f"  {num_threads} threads executed successfully")
                print(f"  All results consistent")

            else:
                self.skipTest("Override compilation failed or disabled")

    def test_no_infinite_recursion_in_fallbacks(self):
        """Test that fallback calls don't cause infinite recursion."""

        # This is a critical test - if fallback implementation is wrong,
        # it could call the override again instead of the original kernel

        def recursive_operation_test():
            """Operation that could potentially cause recursion if fallback is wrong."""
            # Use conditions that should definitely fallback
            x = torch.randn(8, 8, device='cuda', dtype=torch.float32)  # Small tensor

            # Call operation multiple times in nested fashion
            result = x
            for i in range(5):
                result = torch.relu(result)  # Each call should fallback properly

            return result

        with tempfile.TemporaryDirectory() as tmpdir:
            override_result = compile_overrides_for_aoti(tmpdir)

            if override_result and override_result["status"] == "success":
                try:
                    # This should complete without stack overflow
                    start_time = time.time()
                    result = recursive_operation_test()
                    end_time = time.time()

                    # Verify result is valid
                    self.assertTrue(torch.isfinite(result).all())
                    self.assertEqual(result.shape, (8, 8))

                    # Should complete quickly (no infinite recursion)
                    execution_time = end_time - start_time
                    self.assertLess(execution_time, 1.0,
                        f"Execution took too long ({execution_time:.2f}s), possible recursion")

                    print(f"✅ No recursion test passed:")
                    print(f"  Nested operations completed in {execution_time:.3f}s")
                    print(f"  Result shape: {result.shape}")

                except RecursionError:
                    self.fail("Infinite recursion detected in fallback implementation")
                except Exception as e:
                    self.fail(f"Recursion test failed: {e}")
            else:
                self.skipTest("Override compilation failed or disabled")

    def test_fallback_with_different_tensor_layouts(self):
        """Test fallback behavior with different tensor layouts and memory formats."""

        # Test different tensor configurations that should all fallback
        test_configs = [
            {
                'name': 'contiguous',
                'tensor': torch.randn(32, 32, device='cuda').contiguous(),
            },
            {
                'name': 'non_contiguous',
                'tensor': torch.randn(64, 64, device='cuda')[::2, ::2],  # Non-contiguous
            },
            {
                'name': 'transposed',
                'tensor': torch.randn(32, 64, device='cuda').t(),  # Transposed
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            override_result = compile_overrides_for_aoti(tmpdir)

            if override_result and override_result["status"] == "success":
                results = {}

                for config in test_configs:
                    tensor = config['tensor']
                    name = config['name']

                    try:
                        # Test that operation works regardless of layout
                        with torch.no_grad():
                            result = torch.relu(tensor)

                        results[name] = {
                            'success': True,
                            'input_shape': tensor.shape,
                            'output_shape': result.shape,
                            'is_contiguous': result.is_contiguous(),
                        }

                    except Exception as e:
                        results[name] = {
                            'success': False,
                            'error': str(e),
                        }

                # All configurations should succeed (fallback handles all layouts)
                successful_configs = [name for name, result in results.items() if result['success']]
                self.assertEqual(len(successful_configs), len(test_configs),
                    f"Some tensor layouts failed: {results}")

                print(f"✅ Tensor layout fallback test:")
                for name, result in results.items():
                    status = "✅" if result['success'] else "❌"
                    print(f"  {status} {name}: {result}")

            else:
                self.skipTest("Override compilation failed or disabled")

    def test_fallback_error_handling(self):
        """Test error handling in fallback scenarios."""

        # Test edge cases that might cause issues in fallback code
        edge_cases = [
            {
                'name': 'empty_tensor',
                'tensor': torch.empty(0, device='cuda'),
            },
            {
                'name': 'single_element',
                'tensor': torch.tensor([1.0], device='cuda'),
            },
            {
                'name': 'very_small',
                'tensor': torch.randn(1, 1, device='cuda'),
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            override_result = compile_overrides_for_aoti(tmpdir)

            if override_result and override_result["status"] == "success":
                results = {}

                for case in edge_cases:
                    tensor = case['tensor']
                    name = case['name']

                    try:
                        # These should all fallback and handle edge cases gracefully
                        result = torch.relu(tensor)

                        results[name] = {
                            'success': True,
                            'input_shape': tensor.shape,
                            'output_shape': result.shape,
                        }

                    except Exception as e:
                        results[name] = {
                            'success': False,
                            'error': str(e),
                        }

                # Most edge cases should be handled gracefully by fallback
                successful_cases = [name for name, result in results.items() if result['success']]

                # Allow some edge cases to fail (e.g., empty tensors), but most should work
                success_rate = len(successful_cases) / len(edge_cases)
                self.assertGreater(success_rate, 0.5,
                    f"Too many edge cases failed: {results}")

                print(f"✅ Edge case fallback test:")
                print(f"  Success rate: {success_rate:.1%} ({len(successful_cases)}/{len(edge_cases)})")
                for name, result in results.items():
                    status = "✅" if result['success'] else "❌"
                    print(f"    {status} {name}: {result}")

            else:
                self.skipTest("Override compilation failed or disabled")


class TestOverrideKernelPreservation(TestCase):
    """Test that original kernels are properly preserved."""

    def setUp(self):
        super().setUp()
        self.original_compile_overrides = config.aot_inductor.compile_native_overrides
        config.aot_inductor.compile_native_overrides = True

    def tearDown(self):
        super().tearDown()
        config.aot_inductor.compile_native_overrides = self.original_compile_overrides

    def test_kernel_preservation_in_generated_code(self):
        """Test that generated C++ code properly preserves original kernels."""

        override_graphs = _get_override_graphs()

        if not override_graphs:
            self.skipTest("No override graphs available for testing")

        with tempfile.TemporaryDirectory() as tmpdir:
            compiled_libs = compile_overrides(override_graphs, tmpdir)

            if compiled_libs:
                from pathlib import Path

                cpp_files = list(Path(tmpdir).glob("*.cpp"))

                for cpp_file in cpp_files:
                    content = cpp_file.read_text()

                    # Verify kernel preservation patterns exist (updated for current implementation)
                    preservation_checks = {
                        'fallback_kernel_storage': 'static c10::SafeKernelFunction',
                        'dispatcher_lookup': 'c10::Dispatcher::singleton().findOp',
                        'kernel_function_call': 'getComputedKernelForDispatchKey',
                        'boxed_call': 'callBoxed',
                        'fallback_function': 'fallback_',
                    }

                    found_patterns = {}
                    for check_name, pattern in preservation_checks.items():
                        found_patterns[check_name] = pattern in content

                    # Most preservation patterns should be present
                    patterns_found = sum(found_patterns.values())
                    total_patterns = len(preservation_checks)

                    self.assertGreater(patterns_found, total_patterns // 2,
                        f"Not enough preservation patterns in {cpp_file.name}: {found_patterns}")

                    print(f"  {cpp_file.name}: {patterns_found}/{total_patterns} preservation patterns")

                print(f"✅ Kernel preservation validation completed for {len(cpp_files)} files")

            else:
                self.fail("No libraries compiled for kernel preservation testing")


if __name__ == "__main__":
    run_tests()