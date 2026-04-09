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

# Clean up path
sys.path.remove(str(REPO_ROOT))


class TestTritonKernelCompilation(TestCase):
    """Test Triton kernel discovery and compilation for AOTI override system."""

    def test_kernel_discovery_basic(self):
        """Test basic kernel discovery functionality."""
        override_graphs = _get_override_graphs()

        if not override_graphs:
            self.skipTest("No override graphs available in registry")

        triton_kernels = discover_triton_kernels_in_overrides(override_graphs)

        # Should return dict (may be empty if no Triton overrides exist)
        self.assertIsInstance(triton_kernels, dict)

        # If kernels found, validate structure
        for kernel_name, kernel_data in triton_kernels.items():
            self.assertIsInstance(kernel_name, str)
            self.assertIn('source', kernel_data)
            self.assertIn('kernel_name', kernel_data)
            self.assertIn('op_name', kernel_data)
            self.assertIn('dispatch_key', kernel_data)

        print(f"Discovered {len(triton_kernels)} Triton kernels")

    def test_kernel_compilation_pipeline(self):
        """Test end-to-end kernel compilation pipeline."""
        override_graphs = _get_override_graphs()

        if not override_graphs:
            self.skipTest("No override graphs available")

        # Discover kernels
        triton_kernels = discover_triton_kernels_in_overrides(override_graphs)

        if not triton_kernels:
            self.skipTest("No Triton kernels discovered")

        # Compile kernels
        compiled_kernels = _compile_triton_kernels(triton_kernels)

        # Should return dict with compilation results
        self.assertIsInstance(compiled_kernels, dict)

        # Validate compiled kernel structure
        for kernel_name, compiled_data in compiled_kernels.items():
            self.assertIn('kernel_name', compiled_data)
            self.assertIn('source', compiled_data)
            self.assertIn('cached', compiled_data)

        print(f"Compiled {len(compiled_kernels)} Triton kernels")

    def test_aoti_integration_with_kernels(self):
        """Test AOTI integration with Triton kernel compilation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Enable override compilation
            from torch._inductor import config
            original_value = getattr(config.aot_inductor, 'compile_native_overrides', False)

            try:
                config.aot_inductor.compile_native_overrides = True

                # Run AOTI compilation with kernel discovery
                result = compile_overrides_for_aoti(temp_dir)

                if result is None:
                    self.skipTest("Override compilation not enabled or no overrides available")

                # Validate result structure
                self.assertIsInstance(result, dict)
                self.assertIn('status', result)

                if result['status'] == 'success':
                    self.assertIn('statistics', result)
                    stats = result['statistics']

                    # Should include kernel discovery stats
                    self.assertIn('triton_kernels_discovered', stats)
                    self.assertIn('triton_kernels_compiled', stats)

                    print(f"AOTI compilation result: {result}")

            finally:
                config.aot_inductor.compile_native_overrides = original_value

    def test_code_generation_with_kernels(self):
        """Test C++ code generation includes kernel launchers."""
        from torch._inductor.python_native_aoti_code_generation import CodeGenerator
        from torch._inductor.codecache import CudaKernelParamCache

        # Mock a kernel in the cache
        test_kernel_name = "test_silu_CUDA_0_triton"
        original_cache = CudaKernelParamCache.cache.copy()

        try:
            CudaKernelParamCache.cache[test_kernel_name] = {
                'source': '@triton.jit\ndef test_kernel(x): pass',
                'op_name': 'silu',
                'dispatch_key': 'CUDA'
            }

            # Create generator with kernel
            with tempfile.TemporaryDirectory() as temp_dir:
                generator = CodeGenerator(temp_dir, {test_kernel_name: {}})

                # Generate C++ for mock override
                mock_specs = [{
                    'name': 'test_silu_CUDA_0',
                    'op_name': 'silu',
                    'dispatch_key': 'CUDA',
                    'conditions': {'type': 'always_true'},
                    'dsl_name': 'triton'
                }]

                cpp_code = generator._generate_cpp(mock_specs)

                # Verify generated code includes kernel launcher
                self.assertIn(f'{test_kernel_name}_launcher', cpp_code)
                self.assertIn('Generated Triton kernel launchers', cpp_code)

                print("C++ code generation with kernels: PASSED")

        finally:
            CudaKernelParamCache.cache = original_cache


if __name__ == "__main__":
    run_tests()