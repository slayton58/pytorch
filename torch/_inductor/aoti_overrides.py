"""
AOTI integration for compiled PyTorch override dispatch logic.

This module provides hooks for integrating override compilation into the
AOTInductor compilation pipeline, allowing override graphs to be compiled
alongside models for python-less deployment.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from torch._inductor import config

log = logging.getLogger(__name__)


def should_compile_overrides() -> bool:
    """
    Check if override compilation should be enabled.

    Returns:
        True if override compilation is enabled and conditions are met
    """
    # Check configuration flag
    if not getattr(config.aot_inductor, 'compile_native_overrides', False):
        return False

    # Check if we have override graphs to compile
    try:
        import torch._native.registry as registry
        return bool(registry._graphs)
    except ImportError:
        log.debug("torch._native not available, skipping override compilation")
        return False


def compile_overrides_for_aoti(output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Compile override graphs and integrate them into AOTI build using optimized components.

    Args:
        output_dir: Directory where AOTI compilation outputs are being generated

    Returns:
        Dictionary with compilation results, or None if compilation was skipped
    """
    if not should_compile_overrides():
        log.debug("Override compilation disabled or not available")
        return None

    log.info("Compiling native overrides for AOTI...")

    try:
        # Import optimized components
        import sys
        from pathlib import Path

        # Add tools directory to path
        pytorch_root = Path(__file__).parent.parent.parent
        tools_path = pytorch_root / "tools"
        if str(tools_path) not in sys.path:
            sys.path.insert(0, str(tools_path))

        # Use fixed ultra-compressed implementation
        from tools.aoti.integration import compile_overrides_for_aoti as fixed_compile

        # Delegate directly to fixed compressed implementation
        result = fixed_compile(output_dir)

        if result and result.get("status") == "success":
            compiled_libs = result["compiled_libraries"]
            log.info(f"Successfully compiled {len(compiled_libs)} override libraries using compressed pipeline")

            # Return AOTI-compatible format
            return {
                "status": "success",
                "compiled_libraries": compiled_libs,
                "generated_files": list(compiled_libs.values()) if compiled_libs else [],
                "statistics": result.get("statistics", {"generated_overrides": 0})
            }
        elif result:
            log.warning(f"Compressed override compilation: {result.get('status', 'unknown')}")
            return result
        else:
            log.debug("Override compilation skipped")
            return None

    except Exception as e:
        log.error(f"Error during optimized override compilation: {e}", exc_info=True)
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