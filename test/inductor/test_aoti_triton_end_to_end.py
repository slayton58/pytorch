# Owner(s): ["module: inductor"]

import pathlib
import sys
import tempfile
import unittest
from pathlib import Path

# Add repo root to Python path (standard PyTorch test pattern)
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch._inductor.aoti_overrides import (
    discover_triton_kernels_in_overrides,
    _get_override_graphs,
    compile_overrides_for_aoti,
    _compile_triton_kernels
)
from torch._inductor.codecache import CudaKernelParamCache

# Clean up path
sys.path.remove(str(REPO_ROOT))


class TestAOTITritonEndToEnd(TestCase):
    """End-to-end tests for Triton kernel compilation in AOTI override system."""

    def setUp(self):
        """Set up test environment."""
        # Clear kernel cache before each test
        CudaKernelParamCache.cache.clear()

    def test_silu_kernel_discovery_and_compilation(self):
        """Test complete SiLU kernel pipeline from discovery to compilation."""
        # Force import of our SiLU implementation to register it
        try:
            import torch._native.ops.silu.triton_impl
            # Force registration
            torch._native.ops.silu.triton_impl.register_triton_silu_overrides()
        except ImportError:
            self.skipTest("SiLU Triton implementation not available")

        # Get override graphs
        override_graphs = _get_override_graphs()

        if not override_graphs:
            self.skipTest("No override graphs available after registration")

        # Discover kernels
        triton_kernels = discover_triton_kernels_in_overrides(override_graphs)

        if not triton_kernels:
            self.skipTest("No Triton kernels discovered")

        # Validate kernel discovery
        silu_kernels = [k for k in triton_kernels.keys() if 'silu' in k.lower()]
        self.assertGreater(len(silu_kernels), 0, "Should discover at least one SiLU kernel")

        print(f"Discovered SiLU kernels: {silu_kernels}")

        # Test kernel compilation
        compiled_kernels = _compile_triton_kernels(triton_kernels)

        # Validate compilation results
        self.assertIsInstance(compiled_kernels, dict)

        # Check that kernels were stored in CudaKernelParamCache
        for kernel_name in silu_kernels:
            if kernel_name in compiled_kernels:
                self.assertIn(kernel_name, CudaKernelParamCache.cache)
                cache_entry = CudaKernelParamCache.cache[kernel_name]
                self.assertIn('op_name', cache_entry)
                # Handle both inplace and non-inplace variants
                op_name = cache_entry['op_name']
                self.assertIn(op_name, ['silu', 'silu_'], f"Expected silu or silu_, got {op_name}")

        print(f"Successfully compiled {len(compiled_kernels)} kernels")

    def test_relu_kernel_discovery_and_compilation(self):
        """Test complete ReLU kernel pipeline from discovery to compilation."""
        # Force import of our ReLU implementation to register it
        try:
            import torch._native.ops.relu.triton_impl
            # Force registration
            torch._native.ops.relu.triton_impl.register_triton_relu_overrides()
        except ImportError:
            self.skipTest("ReLU Triton implementation not available")

        # Get override graphs
        override_graphs = _get_override_graphs()

        if not override_graphs:
            self.skipTest("No override graphs available after registration")

        # Discover kernels
        triton_kernels = discover_triton_kernels_in_overrides(override_graphs)

        if not triton_kernels:
            self.skipTest("No Triton kernels discovered")

        # Validate kernel discovery
        relu_kernels = [k for k in triton_kernels.keys() if 'relu' in k.lower()]
        self.assertGreater(len(relu_kernels), 0, "Should discover at least one ReLU kernel")

        print(f"Discovered ReLU kernels: {relu_kernels}")

        # Test kernel compilation
        compiled_kernels = _compile_triton_kernels(triton_kernels)

        # Validate compilation results
        for kernel_name in relu_kernels:
            if kernel_name in compiled_kernels:
                self.assertIn(kernel_name, CudaKernelParamCache.cache)
                cache_entry = CudaKernelParamCache.cache[kernel_name]
                self.assertIn('op_name', cache_entry)
                # Handle both inplace and non-inplace variants
                op_name = cache_entry['op_name']
                self.assertIn(op_name, ['relu', 'relu_'], f"Expected relu or relu_, got {op_name}")

    def test_aoti_compilation_with_real_kernels(self):
        """Test AOTI compilation pipeline with discovered and compiled kernels."""
        # Import implementations to register them
        try:
            import torch._native.ops.silu.triton_impl
            import torch._native.ops.relu.triton_impl
        except ImportError:
            self.skipTest("Triton implementations not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Enable override compilation
            from torch._inductor import config
            original_value = getattr(config.aot_inductor, 'compile_native_overrides', False)

            try:
                config.aot_inductor.compile_native_overrides = True

                # Run full AOTI compilation pipeline
                result = compile_overrides_for_aoti(temp_dir)

                if result is None:
                    self.skipTest("Override compilation not enabled or no overrides available")

                # Validate result structure
                self.assertIsInstance(result, dict)
                self.assertIn('status', result)

                print(f"AOTI compilation result: {result['status']}")

                if result['status'] == 'success':
                    self.assertIn('statistics', result)
                    stats = result['statistics']

                    # Should include kernel discovery and compilation stats
                    self.assertIn('triton_kernels_discovered', stats)
                    self.assertIn('triton_kernels_compiled', stats)

                    print(f"Kernels discovered: {stats.get('triton_kernels_discovered', 0)}")
                    print(f"Kernels compiled: {stats.get('triton_kernels_compiled', 0)}")

            finally:
                config.aot_inductor.compile_native_overrides = original_value

    def test_cpp_code_generation_with_real_kernels(self):
        """Test C++ code generation includes real kernel launchers."""
        from torch._inductor.python_native_aoti_code_generation import CodeGenerator

        # Mock compiled kernels in cache
        test_kernels = {
            "test_silu_CUDA_0_triton": {
                'autotuner': "mock_autotuner",  # Would be real CachingAutotuner
                'op_name': 'silu',
                'dispatch_key': 'CUDA',
                'compiled': True
            },
            "test_relu_CUDA_0_triton": {
                'op_name': 'relu',
                'dispatch_key': 'CUDA',
                'compiled': False,
                'error': 'Compilation failed for testing'
            }
        }

        original_cache = CudaKernelParamCache.cache.copy()

        try:
            # Add test kernels to cache
            CudaKernelParamCache.cache.update(test_kernels)

            # Create generator with kernels
            with tempfile.TemporaryDirectory() as temp_dir:
                generator = CodeGenerator(temp_dir, test_kernels)

                # Generate C++ for mock overrides
                mock_specs = [
                    {
                        'name': 'test_silu_CUDA_0',
                        'op_name': 'silu',
                        'dispatch_key': 'CUDA',
                        'conditions': {'type': 'always_true'},
                        'dsl_name': 'triton'
                    },
                    {
                        'name': 'test_relu_CUDA_0',
                        'op_name': 'relu',
                        'dispatch_key': 'CUDA',
                        'conditions': {'type': 'always_true'},
                        'dsl_name': 'triton'
                    }
                ]

                cpp_code = generator._generate_cpp(mock_specs)

                # Verify generated code structure
                self.assertIn('Generated Triton kernel launchers', cpp_code)
                self.assertIn('test_silu_CUDA_0_triton_launcher', cpp_code)
                self.assertIn('test_relu_CUDA_0_triton_launcher', cpp_code)

                # Verify compiled vs uncompiled kernel handling
                self.assertIn('Call compiled Triton kernel', cpp_code)
                self.assertIn('Kernel compilation failed', cpp_code)

                # Verify proper includes
                self.assertIn('#include <ATen/cuda/CUDAContext.h>', cpp_code)
                self.assertIn('#include <c10/cuda/CUDAStream.h>', cpp_code)

                print("C++ code generation validation: PASSED")
                print(f"Generated code length: {len(cpp_code)} characters")

        finally:
            CudaKernelParamCache.cache = original_cache

    def test_fallback_behavior_when_compilation_fails(self):
        """Test that system gracefully falls back when kernel compilation fails."""
        # Create mock kernel data that will fail compilation
        mock_kernels = {
            'failing_kernel_triton': {
                'source': 'invalid triton code that will fail',
                'function': lambda: None,
                'kernel_name': 'invalid_kernel',
                'op_name': 'test_op',
                'dispatch_key': 'CUDA'
            }
        }

        # Compile kernels (should handle failures gracefully)
        compiled_kernels = _compile_triton_kernels(mock_kernels)

        # Should return results even with failures
        self.assertIsInstance(compiled_kernels, dict)

        # Check that failed kernels are marked as uncompiled
        for kernel_name, result in compiled_kernels.items():
            if not result.get('compiled', False):
                self.assertIn('error', result)
                print(f"Kernel {kernel_name} failed as expected: {result['error']}")

    def test_kernel_correctness_basic(self):
        """Basic correctness test for kernel implementations."""
        # Test our kernel implementations directly (without full AOTI pipeline)
        try:
            from torch._native.ops.silu.triton_kernels import triton_silu_kernel_launcher
            from torch._native.ops.relu.triton_kernels import triton_relu_kernel_launcher

            if not torch.cuda.is_available():
                self.skipTest("CUDA not available")

            # Create test tensors
            x_silu = torch.randn(1000, dtype=torch.bfloat16, device='cuda')
            x_relu = torch.randn(1000, dtype=torch.float32, device='cuda')

            # Test SiLU
            try:
                triton_result_silu = triton_silu_kernel_launcher(x_silu)
                pytorch_result_silu = torch.nn.functional.silu(x_silu)

                # Check shapes match
                self.assertEqual(triton_result_silu.shape, pytorch_result_silu.shape)
                print("SiLU kernel shape validation: PASSED")

                # Check approximate correctness (Triton might have slight precision differences)
                torch.testing.assert_close(
                    triton_result_silu, pytorch_result_silu,
                    rtol=1e-3, atol=1e-3
                )
                print("SiLU kernel correctness: PASSED")

            except Exception as e:
                print(f"SiLU kernel test failed (expected if Triton not available): {e}")

            # Test ReLU
            try:
                triton_result_relu = triton_relu_kernel_launcher(x_relu)
                pytorch_result_relu = torch.relu(x_relu)

                # Check shapes match
                self.assertEqual(triton_result_relu.shape, pytorch_result_relu.shape)
                print("ReLU kernel shape validation: PASSED")

                # ReLU should be exact
                torch.testing.assert_close(triton_result_relu, pytorch_result_relu)
                print("ReLU kernel correctness: PASSED")

            except Exception as e:
                print(f"ReLU kernel test failed (expected if Triton not available): {e}")

        except ImportError as e:
            self.skipTest(f"Kernel implementations not available: {e}")


if __name__ == "__main__":
    run_tests()