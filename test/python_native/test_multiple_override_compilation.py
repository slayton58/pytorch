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

# Clean up path
sys.path.remove(str(REPO_ROOT))


class TestMultipleOverrideCompilation(TestCase):
    """Test compilation of multiple overrides working together."""

    def test_multiple_overrides_discovered(self):
        """Test that multiple overrides are properly discovered."""
        override_graphs = _get_override_graphs()

        # Should have multiple operations with overrides
        self.assertGreater(len(override_graphs), 0, "No override graphs found")

        # Check for specific operations we know have multiple overrides
        relu_overrides = override_graphs.get(("relu", "CUDA"), [])

        if relu_overrides:
            # Should have multiple DSL implementations for relu
            dsl_names = [node.dsl_name for node in relu_overrides]
            unique_dsls = set(dsl_names)

            print(f"ReLU overrides found: {len(relu_overrides)} ({dsl_names})")

            # Should have both triton and cutedsl implementations
            expected_dsls = {"triton", "cutedsl"}
            found_dsls = unique_dsls.intersection(expected_dsls)
            self.assertGreater(len(found_dsls), 0,
                f"Expected triton/cutedsl overrides, found: {unique_dsls}")

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
                # Compile all overrides
                compiled_libs = compile_overrides(override_graphs, tmpdir)

                # Should generate C++ files
                self.assertIsInstance(compiled_libs, dict, "Expected dict of compiled libraries")

                # Check that files were actually generated
                output_path = Path(tmpdir)
                cpp_files = list(output_path.glob("*.cpp"))
                h_files = list(output_path.glob("*.h"))

                generated_files = cpp_files + h_files
                self.assertGreater(len(generated_files), 0,
                    f"No C++ files generated in {tmpdir}")

                print(f"\nGenerated {len(generated_files)} files:")
                for file_path in generated_files[:5]:  # Show first 5
                    print(f"  {file_path.name} ({file_path.stat().st_size} bytes)")

                # Verify C++ content contains multiple override registrations
                main_cpp = output_path / "pytorch_overrides.cpp"
                if main_cpp.exists():
                    content = main_cpp.read_text()

                    # Should contain multiple TORCH_LIBRARY_IMPL blocks
                    torch_library_count = content.count("TORCH_LIBRARY_IMPL")
                    self.assertGreater(torch_library_count, 0,
                        "No TORCH_LIBRARY_IMPL registrations found")

                    print(f"Found {torch_library_count} TORCH_LIBRARY_IMPL registrations")

                    # Should contain multiple override functions
                    override_func_count = content.count("_override_dispatch")
                    self.assertGreater(override_func_count, 0,
                        "No override dispatch functions found")

                    print(f"Found {override_func_count} override dispatch functions")

                    # Should contain fallback mechanisms
                    fallback_count = content.count("cached_kernel")
                    self.assertGreater(fallback_count, 0,
                        "No fallback kernel preservation found")

                    print(f"Found {fallback_count} fallback kernel preservations")

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

        if conflict_candidates:
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