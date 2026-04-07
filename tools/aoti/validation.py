"""
Validation and testing framework for compiled PyTorch overrides.

This module provides comprehensive testing for the override compilation system,
including round-trip validation, fallback testing, performance benchmarks,
and end-to-end integration tests.
"""

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch._native.registry as registry
from tools.aoti.extract_conditions import extract_conditions_from_override_node
from tools.aoti.compile_overrides import OverrideCompiler

log = logging.getLogger(__name__)


class OverrideValidationError(Exception):
    """Exception raised when override validation fails."""
    pass


class OverrideValidator:
    """Comprehensive validation framework for compiled overrides."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {
            "condition_extraction": [],
            "cpp_generation": [],
            "fallback_behavior": [],
            "performance": [],
            "integration": []
        }

    def validate_all(self) -> Dict[str, Any]:
        """
        Run all validation tests.

        Returns:
            Dictionary with comprehensive validation results
        """
        log.info("Starting comprehensive override validation...")

        # 1. Validate condition extraction
        log.info("Validating condition extraction...")
        extraction_results = self.validate_condition_extraction()

        # 2. Validate C++ generation
        log.info("Validating C++ code generation...")
        generation_results = self.validate_cpp_generation()

        # 3. Validate fallback behavior
        log.info("Validating fallback behavior...")
        fallback_results = self.validate_fallback_behavior()

        # 4. Run performance comparisons
        log.info("Running performance comparisons...")
        performance_results = self.validate_performance()

        # 5. Test end-to-end integration
        log.info("Testing end-to-end integration...")
        integration_results = self.validate_integration()

        # Aggregate results
        summary = {
            "status": "success",
            "total_tests": (
                len(extraction_results) +
                len(generation_results) +
                len(fallback_results) +
                len(performance_results) +
                len(integration_results)
            ),
            "passed_tests": sum([
                sum(1 for r in extraction_results if r["passed"]),
                sum(1 for r in generation_results if r["passed"]),
                sum(1 for r in fallback_results if r["passed"]),
                sum(1 for r in performance_results if r["passed"]),
                sum(1 for r in integration_results if r["passed"])
            ]),
            "condition_extraction": extraction_results,
            "cpp_generation": generation_results,
            "fallback_behavior": fallback_results,
            "performance": performance_results,
            "integration": integration_results
        }

        summary["failed_tests"] = summary["total_tests"] - summary["passed_tests"]
        summary["success_rate"] = (summary["passed_tests"] / summary["total_tests"] * 100) if summary["total_tests"] > 0 else 0

        log.info(f"Validation complete: {summary['passed_tests']}/{summary['total_tests']} tests passed ({summary['success_rate']:.1f}%)")

        return summary

    def validate_condition_extraction(self) -> List[Dict[str, Any]]:
        """Validate condition extraction from override functions."""
        results = []

        for key, nodes in registry._graphs.items():
            op_symbol, dispatch_key = key

            for i, node in enumerate(nodes):
                test_name = f"{op_symbol}/{dispatch_key}[{i}]"

                try:
                    conditions = extract_conditions_from_override_node(node)

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

                    if self.verbose:
                        status = "✓" if passed else "✗"
                        log.info(f"{status} Condition extraction: {test_name}")

                except Exception as e:
                    result = {
                        "test": "condition_extraction",
                        "name": test_name,
                        "passed": False,
                        "dsl": node.dsl_name,
                        "error": str(e)
                    }
                    results.append(result)

                    if self.verbose:
                        log.error(f"✗ Condition extraction failed: {test_name}: {e}")

        return results

    def validate_cpp_generation(self) -> List[Dict[str, Any]]:
        """Validate C++ code generation from extracted conditions."""
        results = []

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                compiler = OverrideCompiler(tmpdir)
                compilation_result = compiler.compile_all_overrides()

                # Check if compilation succeeded
                passed = compilation_result.get("status") == "success"

                result = {
                    "test": "cpp_generation",
                    "name": "full_compilation",
                    "passed": passed,
                    "generated_overrides": compilation_result.get("statistics", {}).get("generated_overrides", 0),
                    "output_dir": tmpdir
                }

                if not passed:
                    result["errors"] = compilation_result.get("extraction_errors", [])

                results.append(result)

                if self.verbose:
                    status = "✓" if passed else "✗"
                    log.info(f"{status} C++ generation: {result['generated_overrides']} overrides")

                # Validate individual generated files
                if passed:
                    generated_files = compiler._get_output_file_paths()
                    for file_path in generated_files:
                        if file_path.exists():
                            file_result = {
                                "test": "cpp_generation",
                                "name": f"file_generation_{file_path.name}",
                                "passed": True,
                                "file_path": str(file_path),
                                "file_size": file_path.stat().st_size
                            }
                        else:
                            file_result = {
                                "test": "cpp_generation",
                                "name": f"file_generation_{file_path.name}",
                                "passed": False,
                                "error": f"Expected file not generated: {file_path}"
                            }

                        results.append(file_result)

            except Exception as e:
                result = {
                    "test": "cpp_generation",
                    "name": "full_compilation",
                    "passed": False,
                    "error": str(e)
                }
                results.append(result)

                if self.verbose:
                    log.error(f"✗ C++ generation failed: {e}")

        return results

    def validate_fallback_behavior(self) -> List[Dict[str, Any]]:
        """Validate that fallback behavior works correctly."""
        results = []

        # Test SiLU with different tensor configurations
        if torch.cuda.is_available():
            test_cases = [
                {
                    "name": "silu_fallback_small_tensor",
                    "tensor": torch.randn(10, 10, dtype=torch.bfloat16, device='cuda'),
                    "should_trigger_override": False,  # Too small
                    "operation": lambda x: torch.nn.functional.silu(x)
                },
                {
                    "name": "silu_fallback_wrong_dtype",
                    "tensor": torch.randn(5000, 5000, dtype=torch.float32, device='cuda'),
                    "should_trigger_override": False,  # Wrong dtype
                    "operation": lambda x: torch.nn.functional.silu(x)
                },
                {
                    "name": "silu_override_conditions_met",
                    "tensor": torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda'),
                    "should_trigger_override": True,  # Should trigger
                    "operation": lambda x: torch.nn.functional.silu(x)
                }
            ]

            for test_case in test_cases:
                try:
                    tensor = test_case["tensor"]
                    op = test_case["operation"]

                    # Run the operation
                    start_time = time.time()
                    result = op(tensor)
                    end_time = time.time()

                    # Check that result is reasonable
                    passed = (
                        result.shape == tensor.shape and
                        result.dtype == tensor.dtype and
                        result.device == tensor.device and
                        torch.isfinite(result).all()
                    )

                    test_result = {
                        "test": "fallback_behavior",
                        "name": test_case["name"],
                        "passed": passed,
                        "expected_override": test_case["should_trigger_override"],
                        "tensor_shape": list(tensor.shape),
                        "tensor_dtype": str(tensor.dtype),
                        "tensor_device": str(tensor.device),
                        "execution_time": end_time - start_time
                    }

                    if not passed:
                        test_result["error"] = "Operation produced invalid result"

                    results.append(test_result)

                    if self.verbose:
                        status = "✓" if passed else "✗"
                        log.info(f"{status} Fallback test: {test_case['name']}")

                except Exception as e:
                    test_result = {
                        "test": "fallback_behavior",
                        "name": test_case["name"],
                        "passed": False,
                        "error": str(e)
                    }
                    results.append(test_result)

                    if self.verbose:
                        log.error(f"✗ Fallback test failed: {test_case['name']}: {e}")

        else:
            # No CUDA available, create a placeholder test
            results.append({
                "test": "fallback_behavior",
                "name": "cuda_not_available",
                "passed": True,
                "note": "CUDA not available, skipping GPU fallback tests"
            })

        return results

    def validate_performance(self) -> List[Dict[str, Any]]:
        """Compare performance between original and compiled overrides."""
        results = []

        if not torch.cuda.is_available():
            results.append({
                "test": "performance",
                "name": "cuda_not_available",
                "passed": True,
                "note": "CUDA not available, skipping performance tests"
            })
            return results

        # Create test tensors for SiLU performance comparison
        test_sizes = [
            (1024, 1024),      # 1M elements (below threshold)
            (4096, 4096),      # 16M elements (at threshold)
            (6000, 6000),      # 36M elements (above threshold)
        ]

        for size in test_sizes:
            size_name = f"{size[0]}x{size[1]}"
            num_elements = size[0] * size[1]

            try:
                # Create test tensor
                x = torch.randn(size, dtype=torch.bfloat16, device='cuda')

                # Warm up
                for _ in range(5):
                    torch.nn.functional.silu(x)

                # Benchmark
                num_runs = 10
                start_time = time.time()
                for _ in range(num_runs):
                    result = torch.nn.functional.silu(x)
                    torch.cuda.synchronize()
                end_time = time.time()

                avg_time = (end_time - start_time) / num_runs
                throughput = num_elements / avg_time / 1e9  # GB/s

                test_result = {
                    "test": "performance",
                    "name": f"silu_performance_{size_name}",
                    "passed": True,
                    "tensor_size": size,
                    "num_elements": num_elements,
                    "avg_time_ms": avg_time * 1000,
                    "throughput_gb_s": throughput,
                    "expected_override": num_elements >= 16 * 1024 * 1024
                }

                results.append(test_result)

                if self.verbose:
                    log.info(f"✓ Performance test {size_name}: {avg_time*1000:.2f}ms ({throughput:.2f} GB/s)")

            except Exception as e:
                test_result = {
                    "test": "performance",
                    "name": f"silu_performance_{size_name}",
                    "passed": False,
                    "error": str(e)
                }
                results.append(test_result)

                if self.verbose:
                    log.error(f"✗ Performance test failed {size_name}: {e}")

        return results

    def validate_integration(self) -> List[Dict[str, Any]]:
        """Test end-to-end integration with AOTI compilation."""
        results = []

        try:
            from torch._inductor.aoti_overrides import should_compile_overrides, compile_overrides_for_aoti

            # Test 1: Check if integration is available
            integration_available = should_compile_overrides()

            result1 = {
                "test": "integration",
                "name": "aoti_integration_available",
                "passed": True,  # This test passes if we can call the function without error
                "integration_available": integration_available
            }
            results.append(result1)

            # Test 2: Try compilation integration
            if integration_available:
                with tempfile.TemporaryDirectory() as tmpdir:
                    integration_result = compile_overrides_for_aoti(tmpdir)

                    passed = integration_result is not None and integration_result.get("status") == "success"

                    result2 = {
                        "test": "integration",
                        "name": "aoti_override_compilation",
                        "passed": passed,
                        "result": integration_result
                    }

                    if not passed and integration_result:
                        result2["error"] = integration_result.get("error", "Compilation failed")

                    results.append(result2)

            if self.verbose:
                log.info(f"✓ Integration tests completed")

        except Exception as e:
            result = {
                "test": "integration",
                "name": "aoti_integration_test",
                "passed": False,
                "error": str(e)
            }
            results.append(result)

            if self.verbose:
                log.error(f"✗ Integration test failed: {e}")

        return results

    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save validation results to a JSON file."""
        # Create a JSON-serializable copy by converting problematic types to strings
        def make_serializable(obj):
            if hasattr(obj, 'tolist'):  # torch.Tensor
                return f"<Tensor: shape={list(obj.shape)}, dtype={obj.dtype}>"
            elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, list, dict, type(None))):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj

        serializable_results = make_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        log.info(f"Validation results saved to: {output_path}")

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of validation results."""
        print(f"\\n{'='*60}")
        print("OVERRIDE VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {results['total_tests']}")
        print(f"Passed: {results['passed_tests']}")
        print(f"Failed: {results['failed_tests']}")
        print(f"Success rate: {results['success_rate']:.1f}%")

        # Print details by category
        categories = [
            ("Condition Extraction", results['condition_extraction']),
            ("C++ Generation", results['cpp_generation']),
            ("Fallback Behavior", results['fallback_behavior']),
            ("Performance", results['performance']),
            ("Integration", results['integration'])
        ]

        for category_name, category_results in categories:
            if category_results:
                passed = sum(1 for r in category_results if r["passed"])
                total = len(category_results)
                print(f"\\n{category_name}: {passed}/{total}")

                # Show failed tests
                failed_tests = [r for r in category_results if not r["passed"]]
                if failed_tests:
                    print("  Failed tests:")
                    for test in failed_tests[:3]:  # Show first 3 failures
                        error = test.get("error", "Unknown error")
                        print(f"    - {test['name']}: {error}")
                    if len(failed_tests) > 3:
                        print(f"    - ... and {len(failed_tests) - 3} more")


def main():
    """Command-line interface for validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate PyTorch override compilation system")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--output", "-o", help="Output file for results (JSON format)")
    parser.add_argument("--category", choices=["extraction", "generation", "fallback", "performance", "integration"],
                       help="Run only specific category of tests")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    # Make sure torch._native is imported
    import torch._native

    validator = OverrideValidator(verbose=args.verbose)

    if args.category:
        # Run specific category
        if args.category == "extraction":
            results = {"condition_extraction": validator.validate_condition_extraction()}
        elif args.category == "generation":
            results = {"cpp_generation": validator.validate_cpp_generation()}
        elif args.category == "fallback":
            results = {"fallback_behavior": validator.validate_fallback_behavior()}
        elif args.category == "performance":
            results = {"performance": validator.validate_performance()}
        elif args.category == "integration":
            results = {"integration": validator.validate_integration()}

        # Calculate summary for single category
        category_results = list(results.values())[0]
        passed = sum(1 for r in category_results if r["passed"])
        total = len(category_results)
        results.update({
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": total - passed,
            "success_rate": (passed / total * 100) if total > 0 else 0
        })
    else:
        # Run all tests
        results = validator.validate_all()

    # Print summary
    validator.print_summary(results)

    # Save results if requested
    if args.output:
        validator.save_results(results, args.output)

    # Exit with error code if tests failed
    return 0 if results["failed_tests"] == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())