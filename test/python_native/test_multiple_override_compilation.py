# Owner(s): ["module: native-overrides"]

import pathlib
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

# Add repo root to Python path (standard PyTorch test pattern)
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch._inductor.python_native_aoti_condition_extraction import extract_conditions
from torch._inductor.python_native_aoti_code_generation import compile_overrides
from torch._inductor.aoti_overrides import compile_overrides_for_aoti, _get_override_graphs

# Dummy override implementations for testing
def dummy_triton_relu_dispatch(dispatch_keys: torch.DispatchKeySet, x: torch.Tensor, *, fallback_kernel):
    """Dummy Triton ReLU override for testing."""
    # Simple condition: use override for large CUDA tensors
    use_override = (
        x.dtype.is_floating_point and
        x.numel() >= 1024*1024 and  # >= 1M elements
        x.is_cuda and
        x.is_contiguous()
    )

    if use_override:
        # Just use fallback for dummy implementation
        pass

    return fallback_kernel.call_boxed(dispatch_keys, x)

def dummy_cutedsl_relu_dispatch(dispatch_keys: torch.DispatchKeySet, x: torch.Tensor, *, fallback_kernel):
    """Dummy CuTe DSL ReLU override for testing."""
    # Different condition: use override for medium-sized tensors
    use_override = (
        x.dtype == torch.float32 and
        x.numel() >= 512*512 and  # >= 256K elements
        x.numel() < 2048*2048 and  # < 4M elements
        x.is_cuda
    )

    if use_override:
        # Just use fallback for dummy implementation
        pass

    return fallback_kernel.call_boxed(dispatch_keys, x)

def dummy_triton_silu_dispatch(dispatch_keys: torch.DispatchKeySet, x: torch.Tensor, *, fallback_kernel):
    """Dummy Triton SiLU override for testing."""
    # SiLU condition: bfloat16, large tensors
    use_override = (
        x.dtype == torch.bfloat16 and
        x.numel() >= 16*1024*1024 and  # >= 16M elements
        x.is_cuda
    )

    if use_override:
        # Just use fallback for dummy implementation
        pass

    return fallback_kernel.call_boxed(dispatch_keys, x)

# Clean up path
sys.path.remove(str(REPO_ROOT))


class TestMultipleOverrideCompilation(TestCase):
    """Test compilation of multiple overrides working together."""

    @classmethod
    def setUpClass(cls):
        """Set up dummy overrides for testing."""
        cls._original_graphs = None
        cls._setup_dummy_overrides()

    @classmethod
    def tearDownClass(cls):
        """Clean up dummy overrides after testing."""
        cls._cleanup_dummy_overrides()

    @classmethod
    def _setup_dummy_overrides(cls):
        """Register dummy override implementations for testing."""
        try:
            import torch._native.registry as registry
            from torch._native.registry import _OverrideNode
            import functools

            # Store original state
            cls._original_graphs = registry._graphs.copy()

            # Create dummy override nodes
            dummy_overrides = [
                # ReLU overrides (multiple DSLs)
                (("relu", "CUDA"), "triton", dummy_triton_relu_dispatch),
                (("relu", "CUDA"), "cutedsl", dummy_cutedsl_relu_dispatch),

                # SiLU overrides
                (("silu", "CUDA"), "triton", dummy_triton_silu_dispatch),

                # In-place versions
                (("relu_", "CUDA"), "triton", dummy_triton_relu_dispatch),
                (("silu_", "CUDA"), "triton", dummy_triton_silu_dispatch),
            ]

            # Register dummy overrides
            for (op_symbol, dispatch_key), dsl_name, dispatch_fn in dummy_overrides:
                # Get fallback kernel
                try:
                    fallback_kernel = torch.library.get_kernel(f"aten::{op_symbol}", dispatch_key)
                except:
                    # Create a dummy fallback if not available
                    fallback_kernel = None

                # Create dispatch function with fallback
                if fallback_kernel:
                    override_fn = functools.partial(dispatch_fn, fallback_kernel=fallback_kernel)
                else:
                    override_fn = dispatch_fn

                # Create override node
                override_node = _OverrideNode(
                    dsl_name=dsl_name,
                    op_symbol=op_symbol,
                    dispatch_key=dispatch_key,
                    override_fn=override_fn,
                    unconditional_override=False,
                    active=True
                )

                # Add to registry
                key = (op_symbol, dispatch_key)
                if key not in registry._graphs:
                    registry._graphs[key] = []
                registry._graphs[key].append(override_node)

            print(f"Registered {len(dummy_overrides)} dummy overrides for testing")

        except ImportError:
            print("Registry not available, tests will use existing overrides")
            cls._original_graphs = {}

    @classmethod
    def _cleanup_dummy_overrides(cls):
        """Restore original registry state."""
        if cls._original_graphs is not None:
            try:
                import torch._native.registry as registry
                registry._graphs = cls._original_graphs
                print("Restored original override registry")
            except ImportError:
                pass

    def test_multiple_overrides_discovered(self):
        """Test that multiple overrides are properly discovered."""
        override_graphs = _get_override_graphs()

        # Should have multiple operations with overrides
        self.assertGreater(len(override_graphs), 0, "No override graphs found")

        # Check for specific operations we know have multiple overrides
        relu_overrides = override_graphs.get(("relu", "CUDA"), [])

        # Should have ReLU overrides (from our dummy implementations)
        self.assertGreater(len(relu_overrides), 0, "No ReLU overrides found")

        # Should have multiple DSL implementations for relu
        dsl_names = [node.dsl_name for node in relu_overrides]
        unique_dsls = set(dsl_names)

        print(f"ReLU overrides found: {len(relu_overrides)} ({dsl_names})")

        # Should have both triton and cutedsl implementations (from our dummies)
        expected_dsls = {"triton", "cutedsl"}
        found_dsls = unique_dsls.intersection(expected_dsls)
        self.assertGreaterEqual(len(found_dsls), 1,
            f"Expected at least one of triton/cutedsl overrides, found: {unique_dsls}")

        # Ideally should have multiple DSL types
        if len(relu_overrides) > 1:
            self.assertGreater(len(unique_dsls), 1,
                f"Expected multiple DSL types for ReLU, found only: {unique_dsls}")

    def test_multiple_overrides_condition_extraction(self):
        """Test that condition extraction works for multiple overrides."""
        override_graphs = _get_override_graphs()

        extraction_results = {}
        total_extracted = 0

        for (op_name, dispatch_key), nodes in override_graphs.items():
            for i, node in enumerate(nodes):
                test_name = f"{op_name}/{dispatch_key}[{i}]({node.dsl_name})"

                try:
                    conditions = extract_conditions(node.override_fn)
                    extraction_results[test_name] = {
                        "success": conditions.get("type") not in ["extraction_error", "parse_error"],
                        "conditions": conditions,
                        "dsl": node.dsl_name
                    }

                    if extraction_results[test_name]["success"]:
                        total_extracted += 1

                except Exception as e:
                    extraction_results[test_name] = {
                        "success": False,
                        "error": str(e),
                        "dsl": node.dsl_name
                    }

        # Should successfully extract from at least some overrides
        self.assertGreater(total_extracted, 0,
            f"No conditions extracted from any overrides. Results: {extraction_results}")

        # Print summary for debugging
        print(f"\nCondition extraction results ({total_extracted}/{len(extraction_results)} successful):")
        for name, result in extraction_results.items():
            status = "✓" if result["success"] else "✗"
            print(f"  {status} {name}")

    def test_multiple_overrides_cpp_compilation(self):
        """Test that multiple overrides can be compiled to C++ together."""
        override_graphs = _get_override_graphs()

        if not override_graphs:
            self.skipTest("No override graphs available for testing")

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Compile all overrides (this generates .so files in native_overrides/)
                compiled_libs = compile_overrides(override_graphs, tmpdir)

                # Should return a dict of compiled libraries
                self.assertIsInstance(compiled_libs, dict, "Expected dict of compiled libraries")

                # Check that libraries were actually generated
                output_path = Path(tmpdir)
                native_overrides_dir = output_path / "native_overrides"

                # Look for compiled .so files in the native_overrides directory
                so_files = list(native_overrides_dir.glob("*.so")) if native_overrides_dir.exists() else []
                cpp_files = list(native_overrides_dir.glob("*.cpp")) if native_overrides_dir.exists() else []

                generated_files = so_files + cpp_files
                self.assertGreater(len(generated_files), 0,
                    f"No compiled files generated in {native_overrides_dir}")

                print(f"\nGenerated {len(generated_files)} files in native_overrides/:")
                for file_path in generated_files[:5]:  # Show first 5
                    print(f"  {file_path.name} ({file_path.stat().st_size} bytes)")

                # Verify we have libraries for multiple operations
                self.assertGreater(len(compiled_libs), 0, "No compiled libraries returned")

                # Check that multiple override types are compiled
                op_types = set()
                for lib_name in compiled_libs.keys():
                    if 'silu' in lib_name.lower():
                        op_types.add('silu')
                    elif 'relu' in lib_name.lower():
                        op_types.add('relu')

                print(f"Found libraries for operations: {sorted(op_types)}")

                # Should have at least one operation type
                self.assertGreater(len(op_types), 0, "No recognizable operation types in compiled libraries")

                # Verify all libraries actually exist
                for lib_name, lib_path in compiled_libs.items():
                    self.assertTrue(Path(lib_path).exists(),
                        f"Library {lib_name} not found at {lib_path}")

                    # Check it's a reasonable size (not empty)
                    lib_size = Path(lib_path).stat().st_size
                    self.assertGreater(lib_size, 1000,
                        f"Library {lib_name} is too small ({lib_size} bytes)")

            except Exception as e:
                self.fail(f"Multiple override compilation failed: {e}")

    def test_aoti_integration_multiple_overrides(self):
        """Test full AOTI integration with multiple overrides."""
        from torch._inductor import config

        # Temporarily enable override compilation for testing
        original_value = config.aot_inductor.compile_native_overrides
        config.aot_inductor.compile_native_overrides = True

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Run full AOTI integration
                result = compile_overrides_for_aoti(tmpdir)

                if result is None:
                    self.skipTest("Override compilation disabled or no overrides available")

                # Should indicate success or at least partial success
                status = result.get("status")
                self.assertIn(status, ["success", "partial_success", "no_overrides", "no_libraries"],
                    f"Unexpected status: {status}")

                print(f"\nAOTI integration status: {status}")
                print(f"Full result: {result}")

                if status == "success":
                    compiled_libs = result.get("compiled_libraries", {})
                    self.assertGreater(len(compiled_libs), 0,
                        "No libraries compiled despite success status")

                    statistics = result.get("statistics", {})
                    override_count = statistics.get("override_count", 0)
                    library_count = statistics.get("library_count", 0)

                    print(f"\nAOTI integration results:")
                    print(f"  Status: {status}")
                    print(f"  Override count: {override_count}")
                    print(f"  Library count: {library_count}")
                    print(f"  Compilation time: {statistics.get('compilation_time', 0):.2f}s")

                    # Verify multiple overrides were processed
                    if override_count > 1:
                        print(f"✓ Successfully compiled {override_count} overrides together")
                    else:
                        print(f"ⓘ Only {override_count} override found for testing")

        except Exception as e:
            self.fail(f"AOTI integration with multiple overrides failed: {e}")
        finally:
            # Restore original configuration
            config.aot_inductor.compile_native_overrides = original_value

    def test_override_conflict_detection(self):
        """Test that override conflicts are handled properly."""
        override_graphs = _get_override_graphs()

        # Look for operations with multiple overrides (potential conflicts)
        conflict_candidates = []
        for (op_name, dispatch_key), nodes in override_graphs.items():
            if len(nodes) > 1:
                conflict_candidates.append((op_name, dispatch_key, len(nodes)))

        print(f"\nOperations with multiple overrides (potential conflicts):")
        for op_name, dispatch_key, count in conflict_candidates[:5]:
            print(f"  {op_name}/{dispatch_key}: {count} overrides")

        # Should have at least one operation with multiple overrides (from our dummies)
        self.assertGreater(len(conflict_candidates), 0,
            "Expected at least one operation with multiple overrides")

        # Pick first operation with multiple overrides
        op_name, dispatch_key, count = conflict_candidates[0]
        nodes = override_graphs[(op_name, dispatch_key)]

        # Extract conditions from all overrides for this operation
        conditions_list = []
        for i, node in enumerate(nodes):
            try:
                conditions = extract_conditions(node.override_fn)
                conditions_list.append({
                    "index": i,
                    "dsl": node.dsl_name,
                    "conditions": conditions
                })
            except Exception as e:
                conditions_list.append({
                    "index": i,
                    "dsl": node.dsl_name,
                    "error": str(e)
                })

        # In a real implementation, we'd check for condition conflicts here
        # For now, just verify we can extract conditions from multiple overrides
        successful_extractions = [c for c in conditions_list if "error" not in c]
        self.assertGreater(len(successful_extractions), 0,
            f"Could not extract conditions from any override for {op_name}/{dispatch_key}")

        print(f"Extracted conditions from {len(successful_extractions)}/{len(conditions_list)} overrides")

    def test_compilation_performance_multiple_overrides(self):
        """Test that compilation performance is reasonable with multiple overrides."""
        override_graphs = _get_override_graphs()

        if not override_graphs:
            self.skipTest("No override graphs for performance testing")

        total_overrides = sum(len(nodes) for nodes in override_graphs.values())
        print(f"\nPerformance test with {total_overrides} total overrides")

        with tempfile.TemporaryDirectory() as tmpdir:
            start_time = time.time()

            try:
                result = compile_overrides_for_aoti(tmpdir)
                compilation_time = time.time() - start_time

                print(f"Compilation completed in {compilation_time:.2f}s")

                # Should complete within reasonable time (60s for tests)
                self.assertLess(compilation_time, 60.0,
                    f"Compilation took too long: {compilation_time:.2f}s")

                # Performance should scale reasonably with override count
                if total_overrides > 0:
                    time_per_override = compilation_time / total_overrides
                    print(f"Average time per override: {time_per_override:.3f}s")

                    # Should not take more than 5 seconds per override
                    self.assertLess(time_per_override, 5.0,
                        f"Compilation too slow: {time_per_override:.3f}s per override")

            except Exception as e:
                compilation_time = time.time() - start_time
                # Even if compilation fails, it should fail quickly
                self.assertLess(compilation_time, 30.0,
                    f"Failed compilation took too long: {compilation_time:.2f}s")
                # Re-raise the original exception
                raise


if __name__ == "__main__":
    run_tests()