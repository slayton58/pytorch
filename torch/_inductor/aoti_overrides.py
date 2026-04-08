"""
AOTI integration for compiled Python native override dispatch logic.

This module provides hooks for integrating Python native override compilation
into the AOTInductor compilation pipeline, allowing torch._native override
graphs to be compiled alongside models for python-less deployment.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from torch._inductor import config
from .python_native_aoti_condition_extraction import extract_conditions
from .python_native_aoti_code_generation import compile_overrides

log = logging.getLogger(__name__)


def _get_override_graphs():
    """Get override graphs from registry."""
    try:
        import torch._native.registry as registry
        return registry._graphs
    except ImportError:
        return {}


def should_compile_overrides() -> bool:
    """Check if override compilation should be enabled."""
    return (getattr(config.aot_inductor, 'compile_native_overrides', False) and
            _get_override_graphs())


def compile_overrides_for_aoti(output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Main AOTI integration function - compiles overrides for AOTI build pipeline.

    Args:
        output_dir: Directory where AOTI compilation outputs are being generated

    Returns:
        Dictionary with compilation results, or None if compilation was skipped
    """
    # Input validation
    if not output_dir or not isinstance(output_dir, str):
        raise ValueError("output_dir must be a non-empty string")

    if not should_compile_overrides():
        return None

    log.info("Compiling native overrides for AOTI")
    import time
    start_time = time.time()

    try:
        # Get override graphs
        override_graphs = _get_override_graphs()
        if not override_graphs:
            return {"status": "no_overrides", "compiled_libraries": {}}

        # Compile overrides to C++
        compiled_libs = compile_overrides(override_graphs, output_dir)

        compilation_time = time.time() - start_time

        if compiled_libs:
            log.info(f"Successfully compiled {len(compiled_libs)} override libraries in {compilation_time:.2f}s")
            return {
                "status": "success",
                "compiled_libraries": compiled_libs,
                "generated_files": list(compiled_libs.values()),
                "statistics": {
                    "override_count": sum(len(ol) for ol in override_graphs.values()),
                    "library_count": len(compiled_libs),
                    "compilation_time": compilation_time
                }
            }
        else:
            return {
                "status": "no_libraries",
                "compiled_libraries": {},
                "statistics": {"compilation_time": compilation_time}
            }

    except Exception as e:
        log.error(f"Error during override compilation: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "statistics": {"generated_overrides": 0}
        }


def get_override_source_files(aoti_output_dir: str) -> List[str]:
    """
    Get list of generated override C++ source files for AOTI compilation.

    Args:
        aoti_output_dir: AOTI compilation output directory

    Returns:
        List of C++ source file paths to include in compilation
    """
    override_dir = Path(aoti_output_dir) / "native_overrides"

    if not override_dir.exists():
        return []

    source_files = []

    # Main override implementation
    cpp_file = override_dir / "pytorch_overrides.cpp"
    if cpp_file.exists():
        source_files.append(str(cpp_file))

    # Add any additional kernel implementation files
    # TODO: This could be extended to include compiled kernel binaries

    return source_files


def get_override_include_dirs(aoti_output_dir: str) -> List[str]:
    """
    Get list of include directories for override compilation.

    Args:
        aoti_output_dir: AOTI compilation output directory

    Returns:
        List of include directory paths
    """
    override_dir = Path(aoti_output_dir) / "native_overrides"

    if not override_dir.exists():
        return []

    return [str(override_dir)]


def get_override_compile_flags() -> List[str]:
    """
    Get additional compile flags needed for override compilation.

    Returns:
        List of compile flags
    """
    flags = []

    # Add any special flags needed for override compilation
    # For example, if using specific instruction sets or optimizations

    return flags


def integrate_overrides_into_aoti_build(
    aoti_output_dir: str,
    compile_options: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Integrate compiled overrides into AOTI build configuration.

    Args:
        aoti_output_dir: AOTI compilation output directory
        compile_options: Current AOTI compile options

    Returns:
        Updated compile options with override integration
    """
    log.debug("Integrating overrides into AOTI build...")

    # Get override files
    source_files = get_override_source_files(aoti_output_dir)
    include_dirs = get_override_include_dirs(aoti_output_dir)
    compile_flags = get_override_compile_flags()

    if not source_files:
        log.debug("No override source files found, skipping integration")
        return compile_options

    # Update compile options
    updated_options = compile_options.copy()

    # Add source files
    if "additional_sources" not in updated_options:
        updated_options["additional_sources"] = []
    updated_options["additional_sources"].extend(source_files)

    # Add include directories
    if "additional_includes" not in updated_options:
        updated_options["additional_includes"] = []
    updated_options["additional_includes"].extend(include_dirs)

    # Add compile flags
    if "additional_compile_flags" not in updated_options:
        updated_options["additional_compile_flags"] = []
    updated_options["additional_compile_flags"].extend(compile_flags)

    log.info(f"Integrated {len(source_files)} override source files into AOTI build")

    return updated_options


class AOTIOverrideIntegration:
    """Helper class for managing override integration in AOTI compilation."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.override_result: Optional[Dict[str, Any]] = None

    def compile_overrides(self) -> bool:
        """
        Compile overrides and store result.

        Returns:
            True if compilation succeeded, False otherwise
        """
        self.override_result = compile_overrides_for_aoti(str(self.output_dir))

        if self.override_result is None:
            return False

        return self.override_result.get("status") == "success"

    def get_integration_info(self) -> Dict[str, Any]:
        """
        Get information about override integration for AOTI compilation.

        Returns:
            Dictionary with integration information
        """
        if not self.override_result:
            return {"enabled": False}

        return {
            "enabled": True,
            "status": self.override_result.get("status"),
            "generated_files": self.override_result.get("generated_files", []),
            "statistics": self.override_result.get("statistics", {}),
            "source_files": get_override_source_files(str(self.output_dir)),
            "include_dirs": get_override_include_dirs(str(self.output_dir)),
            "compile_flags": get_override_compile_flags()
        }

    def update_compile_options(self, compile_options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update AOTI compile options to include override integration.

        Args:
            compile_options: Current compile options

        Returns:
            Updated compile options
        """
        if not self.override_result or self.override_result.get("status") != "success":
            return compile_options

        return integrate_overrides_into_aoti_build(str(self.output_dir), compile_options)