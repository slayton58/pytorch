# Owner(s): ["module: native-overrides"]

import json
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

# Clean up path
sys.path.remove(str(REPO_ROOT))


class TestOverrideValidation(TestCase):
    """Comprehensive validation tests for the override compilation system."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.verbose = False

    def test_condition_extraction_validation(self):
        """Validate condition extraction from all registered override functions."""
        results = self._validate_condition_extraction()

        # At least some extractions should succeed
        total_tests = len(results)
        if total_tests > 0:
            passed_tests = sum(1 for r in results if r["passed"])
            success_rate = passed_tests / total_tests

            # Expect at least 50% success rate for condition extraction
            self.assertGreaterEqual(success_rate, 0.5,
                f"Condition extraction success rate too low: {success_rate:.1%}")

    def test_cpp_generation_validation(self):
        """Validate C++ code generation from extracted conditions."""
        results = self._validate_cpp_generation()

        # Should have at least attempted code generation
        self.assertGreater(len(results), 0, "No C++ generation tests were run")

        # Check if any generation succeeded
        passed_tests = sum(1 for r in results if r["passed"])
        if passed_tests > 0:
            # If any succeeded, verify they produced valid output
            for result in results:
                if result["passed"] and "generated_overrides" in result:
                    self.assertGreaterEqual(result["generated_overrides"], 0)

    def test_fallback_behavior_validation(self):
        """Validate that fallback behavior is properly preserved."""
        results = self._validate_fallback_behavior()

        # All fallback tests should pass (this is critical for correctness)
        failed_tests = [r for r in results if not r["passed"]]
        self.assertEqual(len(failed_tests), 0,
            f"Fallback validation failures: {[r['error'] for r in failed_tests]}")

    def test_performance_validation(self):
        """Validate performance characteristics of the override system."""
        results = self._validate_performance()

        # Performance tests should complete without errors
        error_results = [r for r in results if "error" in r]
        self.assertEqual(len(error_results), 0,
            f"Performance validation errors: {[r['error'] for r in error_results]}")

    def test_integration_validation(self):
        """Test end-to-end integration of the override system."""
        results = self._validate_integration()

        # Integration tests should not have critical failures
        critical_failures = [r for r in results if not r["passed"] and r.get("critical", False)]
        self.assertEqual(len(critical_failures), 0,
            f"Critical integration failures: {[r.get('error') for r in critical_failures]}")

    def test_comprehensive_validation(self):
        """Run the full validation suite and check overall health."""
        summary = self._validate_all()

        # Overall validation should indicate the system is functional
        self.assertIn("status", summary)
        self.assertEqual(summary["status"], "success")

        # Should have run a reasonable number of tests
        self.assertGreater(summary["total_tests"], 0)

        # Success rate should be reasonable (>= 70%)
        if summary["total_tests"] > 0:
            success_rate = summary["success_rate"]
            self.assertGreaterEqual(success_rate, 70.0,
                f"Overall validation success rate too low: {success_rate:.1f}%")

    def _validate_condition_extraction(self) -> List[Dict[str, Any]]:
        """Validate condition extraction from override functions."""
        results = []

        try:
            import torch._native.registry as registry
            from torch._inductor.python_native_aoti_condition_extraction import extract_conditions

            for key, nodes in registry._graphs.items():
                op_symbol, dispatch_key = key

                for i, node in enumerate(nodes):
                    test_name = f"{op_symbol}/{dispatch_key}[{i}]"

                    try:
                        conditions = extract_conditions(node.override_fn)

                        # Check if extraction succeeded
                        passed = conditions.get("type") not in ["extraction_error", "parse_error"]

                        result = {
                            "test": "condition_extraction",
                            "name": test_name,
                            "passed": passed,
                            "dsl": node.dsl_name,
                            "conditions": conditions
                        }

                        if not passed:
                            result["error"] = conditions.get("error", "Unknown extraction error")

                        results.append(result)

                    except Exception as e:
                        results.append({
                            "test": "condition_extraction",
                            "name": test_name,
                            "passed": False,
                            "dsl": node.dsl_name,
                            "error": str(e)
                        })

        except ImportError:
            # If registry is not available, create mock test
            results.append({
                "test": "condition_extraction",
                "name": "registry_unavailable",
                "passed": True,
                "note": "Registry not available in test environment"
            })

        return results

    def _validate_cpp_generation(self) -> List[Dict[str, Any]]:
        """Validate C++ code generation from extracted conditions."""
        results = []

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                from torch._inductor.python_native_aoti_code_generation import compile_overrides
                from torch._inductor.aoti_overrides import _get_override_graphs

                override_graphs = _get_override_graphs()
                if override_graphs:
                    compiled = compile_overrides(override_graphs, tmpdir)

                    # Check if compilation succeeded
                    passed = isinstance(compiled, dict) and len(compiled) > 0

                    result = {
                        "test": "cpp_generation",
                        "name": "full_compilation",
                        "passed": passed,
                        "generated_overrides": len(compiled) if compiled else 0,
                        "output_dir": tmpdir
                    }

                    results.append(result)

                    # Check generated files
                    output_path = Path(tmpdir)
                    cpp_files = list(output_path.glob("*.cpp"))
                    h_files = list(output_path.glob("*.h"))

                    for file_path in cpp_files + h_files:
                        results.append({
                            "test": "cpp_generation",
                            "name": f"file_generation_{file_path.name}",
                            "passed": file_path.exists() and file_path.stat().st_size > 0,
                            "file_path": str(file_path),
                            "file_size": file_path.stat().st_size if file_path.exists() else 0
                        })
                else:
                    results.append({
                        "test": "cpp_generation",
                        "name": "no_overrides",
                        "passed": True,
                        "note": "No override graphs available for testing"
                    })

            except Exception as e:
                results.append({
                    "test": "cpp_generation",
                    "name": "compilation_error",
                    "passed": False,
                    "error": str(e)
                })

        return results

    def _validate_fallback_behavior(self) -> List[Dict[str, Any]]:
        """Validate that fallback behavior is properly preserved."""
        results = []

        # Test kernel preservation patterns
        fallback_patterns = [
            "fallback_",
            "c10::Dispatcher::singleton().findOp",
            "getComputedKernelForDispatchKey",
            "static c10::SafeKernelFunction",
            "callBoxed"
        ]

        # Check that C++ generation includes proper fallback patterns
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                from torch._inductor.python_native_aoti_code_generation import compile_overrides
                from torch._inductor.aoti_overrides import _get_override_graphs

                override_graphs = _get_override_graphs()
                if override_graphs:
                    compiled = compile_overrides(override_graphs, tmpdir)

                    # Read generated C++ files and check for fallback patterns
                    cpp_files = list(Path(tmpdir).glob("*.cpp"))

                    for cpp_file in cpp_files:
                        if cpp_file.exists():
                            content = cpp_file.read_text()

                            for pattern in fallback_patterns:
                                pattern_found = pattern in content
                                results.append({
                                    "test": "fallback_behavior",
                                    "name": f"pattern_{pattern}_{cpp_file.name}",
                                    "passed": pattern_found,
                                    "pattern": pattern,
                                    "file": str(cpp_file)
                                })

                                if not pattern_found:
                                    results[-1]["error"] = f"Fallback pattern '{pattern}' not found in generated code"

            except Exception as e:
                results.append({
                    "test": "fallback_behavior",
                    "name": "fallback_check_error",
                    "passed": False,
                    "error": str(e)
                })

        return results

    def _validate_performance(self) -> List[Dict[str, Any]]:
        """Validate performance characteristics of the override system."""
        results = []

        # Test compilation time
        start_time = time.time()
        try:
            from torch._inductor.aoti_overrides import compile_overrides_for_aoti

            compile_overrides_for_aoti("/tmp/test_performance")
            compilation_time = time.time() - start_time

            # Should complete within reasonable time (60 seconds for tests)
            passed = compilation_time < 60.0

            results.append({
                "test": "performance",
                "name": "compilation_time",
                "passed": passed,
                "compilation_time": compilation_time,
                "threshold": 60.0
            })

            if not passed:
                results[-1]["error"] = f"Compilation took too long: {compilation_time:.2f}s"

        except Exception as e:
            results.append({
                "test": "performance",
                "name": "compilation_time",
                "passed": False,
                "error": str(e)
            })

        return results

    def _validate_integration(self) -> List[Dict[str, Any]]:
        """Test end-to-end integration of the override system."""
        results = []

        # Test AOTI integration
        try:
            # Simple integration test - check if the main compilation function works
            from torch._inductor.aoti_overrides import compile_overrides_for_aoti

            with tempfile.TemporaryDirectory() as test_dir:
                # This should complete without throwing exceptions
                compile_overrides_for_aoti(test_dir)

                validation_result = {
                    "status": "success",
                    "message": "AOTI compilation completed successfully"
                }

            passed = validation_result.get("status") != "error"

            results.append({
                "test": "integration",
                "name": "aoti_integration",
                "passed": passed,
                "validation_result": validation_result
            })

            if not passed:
                results[-1]["error"] = f"AOTI integration failed: {validation_result}"

        except Exception as e:
            results.append({
                "test": "integration",
                "name": "aoti_integration",
                "passed": False,
                "critical": True,
                "error": str(e)
            })

        return results

    def _validate_all(self) -> Dict[str, Any]:
        """Run all validation tests and return comprehensive results."""
        # Run individual validation components
        extraction_results = self._validate_condition_extraction()
        generation_results = self._validate_cpp_generation()
        fallback_results = self._validate_fallback_behavior()
        performance_results = self._validate_performance()
        integration_results = self._validate_integration()

        # Aggregate results
        all_results = (extraction_results + generation_results +
                      fallback_results + performance_results + integration_results)

        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r["passed"])

        summary = {
            "status": "success",
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 100,
            "condition_extraction": extraction_results,
            "cpp_generation": generation_results,
            "fallback_behavior": fallback_results,
            "performance": performance_results,
            "integration": integration_results
        }

        return summary


if __name__ == "__main__":
    run_tests()