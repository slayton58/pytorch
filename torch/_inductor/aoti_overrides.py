"""
AOTI integration for compiled Python native override dispatch logic.

This module provides hooks for integrating Python native override compilation
into the AOTInductor compilation pipeline, allowing torch._native override
graphs to be compiled alongside models for python-less deployment.
"""

import logging
import os
import tempfile
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from torch._inductor import config
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.codecache import CudaKernelParamCache
from .python_native_aoti_condition_extraction import extract_conditions
from .python_native_aoti_code_generation import compile_overrides

log = logging.getLogger(__name__)


def _get_override_graphs():
    """Get override graphs from registry."""
    try:
        import torch._native.registry as registry

        # Ensure override modules are imported to trigger registration
        _import_known_override_modules()

        return registry._graphs
    except ImportError:
        return {}


def _import_known_override_modules():
    """Import known override modules to ensure registration occurs."""
    override_modules = [
        'torch._native.ops.silu.triton_impl',
        'torch._native.ops.relu.triton_impl',
    ]

    for module_name in override_modules:
        try:
            importlib.import_module(module_name)
            log.debug(f"Imported override module: {module_name}")
        except ImportError as e:
            log.debug(f"Failed to import override module {module_name}: {e}")


def should_compile_overrides() -> bool:
    """Check if override compilation should be enabled."""
    return (getattr(config.aot_inductor, 'compile_native_overrides', False) and
            _get_override_graphs())


def discover_triton_kernels_in_overrides(override_graphs: Dict) -> Dict[str, Dict]:
    """
    Find Triton kernels in override implementations using existing PyTorch utilities.

    Args:
        override_graphs: Dict from torch._native.registry._graphs

    Returns:
        Dict mapping kernel_name -> kernel_data with source, metadata, etc.
    """
    discovered_kernels = {}

    for (op_name, dispatch_key), override_list in override_graphs.items():
        for i, override_node in enumerate(override_list):
            # Only process Triton DSL overrides
            if hasattr(override_node, 'dsl_name') and override_node.dsl_name == 'triton':
                try:
                    # Extract kernel info from the override implementation
                    kernel_info = _extract_triton_kernel_from_override(override_node, op_name, dispatch_key, i)
                    if kernel_info:
                        kernel_name = f"{op_name}_{dispatch_key}_{i}_triton"
                        discovered_kernels[kernel_name] = kernel_info
                        log.info(f"Discovered Triton kernel: {kernel_name}")
                except Exception as e:
                    log.warning(f"Failed to extract Triton kernel from {op_name}_{dispatch_key}_{i}: {e}")

    return discovered_kernels


def _extract_triton_kernel_from_override(override_node, op_name: str, dispatch_key: str, index: int) -> Optional[Dict]:
    """
    Extract Triton kernel information from a single override node.

    Args:
        override_node: Override node from registry
        op_name: Operation name (e.g. 'silu')
        dispatch_key: Dispatch key (e.g. 'CUDA')
        index: Index in override list

    Returns:
        Dict with kernel source, metadata, etc. or None if no kernel found
    """
    try:
        # Get the module path from the override function
        override_fn = override_node.override_fn

        # Handle functools.partial objects
        if hasattr(override_fn, 'func'):
            # This is a functools.partial - get the wrapped function
            actual_fn = override_fn.func
            if not hasattr(actual_fn, '__module__'):
                return None
            override_module_path = actual_fn.__module__
        elif hasattr(override_fn, '__module__'):
            override_module_path = override_fn.__module__
        else:
            return None

        # Look for corresponding triton_kernels module

        # Convert torch._native.ops.silu.triton_impl -> torch._native.ops.silu.triton_kernels
        if '.triton_impl' in override_module_path:
            kernel_module_path = override_module_path.replace('.triton_impl', '.triton_kernels')
        else:
            # Fallback: assume triton_kernels is in same package
            base_path = '.'.join(override_module_path.split('.')[:-1])
            kernel_module_path = f"{base_path}.triton_kernels"

        # Try to import the triton_kernels module
        try:
            kernel_module = importlib.import_module(kernel_module_path)
        except ImportError:
            log.debug(f"No triton_kernels module found at {kernel_module_path}")
            return None

        # Find @triton.jit decorated functions in the module
        triton_kernels = _find_triton_jit_functions(kernel_module)

        if not triton_kernels:
            log.debug(f"No @triton.jit functions found in {kernel_module_path}")
            return None

        # For now, take the first Triton kernel found
        # TODO: Could be enhanced to match based on function name patterns
        kernel_info = triton_kernels[0]

        source_code = kernel_info['source']
        kernel_name = kernel_info['name']
        kernel_fn = kernel_info['function']  # May be None for AST-discovered functions

        # Basic metadata extraction
        # TODO: Could be enhanced to use Inductor's signature_to_meta utilities
        kernel_result = {
            'source': source_code,
            'function': kernel_fn,
            'module_path': kernel_module_path,
            'kernel_name': kernel_name,
            'override_node': override_node,
            'op_name': op_name,
            'dispatch_key': dispatch_key,
            'discovery_method': kernel_info['method']
        }

        return kernel_result

    except Exception as e:
        log.debug(f"Failed to extract kernel from override: {e}")
        return None


def _find_triton_jit_functions(module) -> List:
    """
    Find functions decorated with @triton.jit in a module.
    Handles conditionally defined functions inside `if HAS_TRITON:` blocks.

    Args:
        module: Python module to scan

    Returns:
        List of Triton kernel data with source code and metadata
    """
    triton_kernels = []

    # Method 1: Look for functions currently defined in module
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        try:
            source = inspect.getsource(obj)
            if source and ('@triton.jit' in source or 'triton.jit' in source):
                triton_kernels.append({
                    'function': obj,
                    'name': name,
                    'source': source,
                    'method': 'runtime_discovery'
                })
                continue
        except OSError:
            pass

    # Method 2: Parse module source to find @triton.jit functions that may be conditionally defined
    try:
        import ast

        # Get the module's source file
        module_file = inspect.getfile(module)
        with open(module_file, 'r') as f:
            module_source = f.read()

        # Parse the AST to find @triton.jit decorated functions
        tree = ast.parse(module_source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function has @triton.jit decorator
                has_triton_jit = False
                for decorator in node.decorator_list:
                    decorator_source = ast.unparse(decorator) if hasattr(ast, 'unparse') else str(decorator)
                    if 'triton.jit' in decorator_source or '@triton.jit' in decorator_source:
                        has_triton_jit = True
                        break

                if has_triton_jit:
                    # Extract the function source INCLUDING decorators from the original module source
                    # We need to find the start of the decorators, not just the function definition
                    all_lines = module_source.split('\n')

                    # Find the line with the first decorator for this function
                    start_line = node.lineno - 1  # Convert to 0-based indexing

                    # Look backwards to find any decorators that belong to this function
                    decorator_start = start_line
                    if node.decorator_list:
                        # If there are decorators, find the first one
                        first_decorator_line = node.decorator_list[0].lineno - 1
                        decorator_start = first_decorator_line

                    # Extract from decorator start to function end
                    func_lines = all_lines[decorator_start:node.end_lineno]

                    # Find the correct indentation by looking at the first decorator or function line
                    if func_lines:
                        # Get indentation from the first line (decorator or function)
                        first_line = func_lines[0]
                        indent = len(first_line) - len(first_line.lstrip())

                        # Remove common indentation from all lines
                        func_source_lines = []
                        for line in func_lines:
                            if line.strip():  # Skip empty lines
                                func_source_lines.append(line[indent:] if len(line) > indent else line)
                            else:
                                func_source_lines.append('')

                        # Make the kernel source self-contained by adding necessary imports
                        kernel_imports = [
                            "import triton",
                            "import triton.language as tl"
                        ]

                        # Combine imports with the function source
                        full_source = '\n'.join(kernel_imports) + '\n\n' + '\n'.join(func_source_lines)

                        triton_kernels.append({
                            'function': None,  # Function may not be accessible at runtime
                            'name': node.name,
                            'source': full_source,
                            'method': 'ast_parsing'
                        })

    except Exception as e:
        log.debug(f"AST parsing failed for {module}: {e}")

    # Remove duplicates (same function found by both methods)
    seen_names = set()
    unique_kernels = []
    for kernel in triton_kernels:
        if kernel['name'] not in seen_names:
            seen_names.add(kernel['name'])
            unique_kernels.append(kernel)

    return unique_kernels


def _extract_cubin_from_jit_function(jit_function, kernel_name: str) -> tuple:
    """
    Extract CUBIN binary data from a compiled Triton JITFunction.

    Args:
        jit_function: Triton JITFunction object
        kernel_name: Name for the kernel (for temp file naming)

    Returns:
        (cubin_data: bytes, cubin_path: str) or (None, None) if extraction fails
    """
    try:
        import tempfile
        import triton

        log.info(f"Extracting CUBIN from JIT function: {kernel_name}")

        # Force compilation by calling the kernel with dummy arguments
        # This ensures the CUBIN is actually compiled and cached

        # Create dummy arguments to force compilation
        # Most kernels expect: x_ptr, output_ptr, n_elements, BLOCK_SIZE
        dummy_n_elements = 1024

        # Create dummy CUDA tensors with proper types
        dummy_input = torch.zeros(dummy_n_elements, device='cuda', dtype=torch.bfloat16)
        dummy_output = torch.zeros(dummy_n_elements, device='cuda', dtype=torch.bfloat16)

        # Calculate grid size
        BLOCK_SIZE = 1024
        grid_size = (dummy_n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Force compilation by calling with correct Triton calling convention
        try:
            # Triton kernels are called with grid as first argument, then kernel args
            jit_function[(grid_size,)](
                dummy_input,        # Pass tensor, not data_ptr()
                dummy_output,       # Pass tensor, not data_ptr()
                dummy_n_elements,
                BLOCK_SIZE=BLOCK_SIZE
            )
        except Exception as e:
            log.warning(f"Failed to force compile {kernel_name} with tensors: {e}")
            # Try with data pointers and proper types
            try:
                jit_function[(grid_size,)](
                    dummy_input.data_ptr(),
                    dummy_output.data_ptr(),
                    dummy_n_elements,
                    BLOCK_SIZE=BLOCK_SIZE
                )
            except Exception as e2:
                log.warning(f"Failed to force compile {kernel_name} with pointers: {e2}")
                # Try calling without forcing compilation - just access attributes
                pass

        # Access the compiled binary from Triton's device_caches
        if hasattr(jit_function, 'device_caches') and jit_function.device_caches:
            # Get the first device cache (usually device 0)
            for device_id, cache_data in jit_function.device_caches.items():
                if len(cache_data) > 0:
                    kernel_cache = cache_data[0]  # First element is the kernel cache dict

                    # Get the first (and likely only) compiled kernel
                    for cache_key, compiled_kernel in kernel_cache.items():
                        if hasattr(compiled_kernel, 'asm') and hasattr(compiled_kernel.asm, 'get'):
                            cubin_data = compiled_kernel.asm.get('cubin')

                            if cubin_data and isinstance(cubin_data, bytes):
                                # Write CUBIN to temporary file
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.cubin', prefix=f'{kernel_name}_') as f:
                                    f.write(cubin_data)
                                    cubin_path = f.name

                                log.info(f"Extracted CUBIN for {kernel_name}: {len(cubin_data)} bytes -> {cubin_path}")
                                return cubin_data, cubin_path

        log.warning(f"Could not extract CUBIN from {kernel_name} - no binary data found")
        return None, None

    except Exception as e:
        log.warning(f"CUBIN extraction failed for {kernel_name}: {e}")
        return None, None


def _compile_triton_kernels(triton_kernels: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Compile discovered Triton kernels using PyTorch Inductor's existing infrastructure.

    Args:
        triton_kernels: Dict from discover_triton_kernels_in_overrides()

    Returns:
        Dict mapping kernel_name -> compiled kernel info with real CachingAutotuners
    """
    if not triton_kernels:
        return {}

    # Import PyTorch's Triton compilation infrastructure
    try:
        from torch._inductor.codecache import _load_triton_kernel_from_source
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    except ImportError as e:
        log.warning(f"PyTorch Triton infrastructure not available: {e}")
        return {}

    compiled_kernels = {}

    for kernel_name, kernel_data in triton_kernels.items():
        try:
            source_code = kernel_data['source']
            kernel_function = kernel_data['function']
            actual_kernel_name = kernel_data['kernel_name']

            log.info(f"Compiling Triton kernel: {kernel_name}")

            # Use PyTorch's real Triton compilation infrastructure
            try:
                # Load kernel from source using PyTorch's existing system
                # This compiles the Triton code to CUBIN and creates a CachingAutotuner
                autotuner = _load_triton_kernel_from_source(
                    kernel_name=actual_kernel_name,
                    source_code=source_code
                )

                # Handle different return types from Triton compilation
                if hasattr(autotuner, '__class__') and 'JITFunction' in str(type(autotuner)):
                    # Got a JITFunction - this is a successfully compiled Triton kernel
                    log.info(f"Got JITFunction for {kernel_name} - extracting CUBIN")

                    # Force compilation and extract CUBIN binary
                    cubin_data, cubin_path = _extract_cubin_from_jit_function(autotuner, kernel_name)

                    # Create a wrapper that provides autotuner-like interface with CUBIN data
                    class TritonJITWrapper:
                        def __init__(self, jit_function, cubin_path=None):
                            self.jit_function = jit_function
                            self.cache_file_path = cubin_path
                            self.num_warps = 4
                            self.num_stages = 2

                    autotuner = TritonJITWrapper(autotuner, cubin_path)

                elif not isinstance(autotuner, CachingAutotuner):
                    log.warning(f"Expected CachingAutotuner or JITFunction, got {type(autotuner)} for {kernel_name}")
                    continue

                # Store compiled kernel in existing CudaKernelParamCache used by Inductor
                CudaKernelParamCache.cache[kernel_name] = {
                    'autotuner': autotuner,
                    'source': source_code,
                    'kernel_function': kernel_function,
                    'op_name': kernel_data['op_name'],
                    'dispatch_key': kernel_data['dispatch_key'],
                    'actual_kernel_name': actual_kernel_name,
                    # Extract metadata from autotuner
                    'num_warps': getattr(autotuner, 'num_warps', 4),
                    'num_stages': getattr(autotuner, 'num_stages', 2),
                    # Store CUBIN path if available
                    'cubin_path': getattr(autotuner, 'cache_file_path', None),
                    'compiled': True,
                    'jit_function': getattr(autotuner, 'jit_function', None)
                }

                compiled_kernels[kernel_name] = {
                    'kernel_name': kernel_name,
                    'autotuner': autotuner,
                    'source': source_code,
                    'compiled': True
                }

                log.info(f"Successfully compiled Triton kernel: {kernel_name}")

            except Exception as e:
                log.warning(f"Failed to compile Triton kernel {kernel_name}: {e}")

                # Store as uncompiled for fallback behavior
                CudaKernelParamCache.cache[kernel_name] = {
                    'source': source_code,
                    'kernel_function': kernel_function,
                    'op_name': kernel_data['op_name'],
                    'dispatch_key': kernel_data['dispatch_key'],
                    'compiled': False,
                    'error': str(e)
                }

                compiled_kernels[kernel_name] = {
                    'kernel_name': kernel_name,
                    'source': source_code,
                    'compiled': False,
                    'error': str(e)
                }

        except Exception as e:
            log.warning(f"Error processing Triton kernel {kernel_name}: {e}")
            continue

    log.info(f"Compiled {len([k for k, v in compiled_kernels.items() if v.get('compiled', False)])} out of {len(triton_kernels)} Triton kernels")
    return compiled_kernels


def _create_runtime_bridge(compiled_kernels: Dict) -> bool:
    """
    Create runtime bridge for C++ access to compiled Triton kernels.

    Args:
        compiled_kernels: Dict of compiled kernel results

    Returns:
        True if runtime bridge created successfully
    """
    try:
        from .aoti_triton_runtime import create_kernel_cache_bridge

        # Initialize the runtime system and register kernels
        bridge_success = create_kernel_cache_bridge()

        if bridge_success:
            log.info("Successfully created runtime bridge for C++ kernel access")
            return True
        else:
            log.warning("Failed to create complete runtime bridge")
            return False

    except ImportError as e:
        log.warning(f"Runtime bridge not available: {e}")
        return False
    except Exception as e:
        log.error(f"Error creating runtime bridge: {e}")
        return False


def _setup_runtime_library_loading(output_dir: str, compiled_libs: Dict) -> bool:
    """
    Set up runtime library loading for compiled override libraries.

    Args:
        output_dir: AOTI compilation output directory
        compiled_libs: Dict of compiled library paths

    Returns:
        True if runtime loading setup succeeded
    """
    try:
        from .aoti_library_loader import (
            setup_runtime_library_loading,
            create_cmake_integration
        )

        # Setup library loading
        setup_success = setup_runtime_library_loading(output_dir)

        # Create CMake integration for external projects
        cmake_success = create_cmake_integration(output_dir)

        if setup_success and cmake_success:
            log.info("Runtime library loading fully configured")
            return True
        elif setup_success:
            log.info("Runtime library loading configured (CMake integration failed)")
            return True
        else:
            log.warning("Runtime library loading setup incomplete")
            return False

    except ImportError as e:
        log.warning(f"Library loading system not available: {e}")
        return False
    except Exception as e:
        log.error(f"Error setting up runtime library loading: {e}")
        return False


def embed_override_cubins(aoti_output_dir: str, compiled_kernels: Dict) -> List[str]:
    """
    Embed override CUBIN binaries using PyTorch's existing binary embedding system.

    Args:
        aoti_output_dir: AOTI compilation output directory
        compiled_kernels: Dict of compiled kernel results

    Returns:
        List of generated object file paths
    """
    if not compiled_kernels:
        return []

    try:
        from torch._inductor.codecache import batch_convert_cubins_to_obj
        import shutil
    except ImportError:
        log.warning("CUBIN embedding infrastructure not available")
        return []

    # Collect CUBIN paths from successfully compiled kernels
    cubin_tuples = []  # List of (cubin_path, kernel_name) tuples

    for kernel_name, kernel_data in compiled_kernels.items():
        if kernel_data.get('compiled', False):
            # Get CUBIN path from CudaKernelParamCache
            cache_entry = CudaKernelParamCache.cache.get(kernel_name, {})
            cubin_path = cache_entry.get('cubin_path')

            if cubin_path and Path(cubin_path).exists():
                cubin_tuples.append((cubin_path, kernel_name))
                log.info(f"Found CUBIN for embedding: {kernel_name} -> {cubin_path}")

    if not cubin_tuples:
        log.info("No CUBIN files found for embedding")
        return []

    try:
        # Use PyTorch's existing binary embedding system
        log.info(f"Embedding {len(cubin_tuples)} CUBIN files using PyTorch infrastructure")

        # Create temporary directory for object file generation
        import tempfile
        with tempfile.TemporaryDirectory() as temp_obj_dir:
            obj_file_path = batch_convert_cubins_to_obj(cubin_tuples, temp_obj_dir)

            # Copy the single combined object file to override directory for linker
            override_dir = Path(aoti_output_dir) / "native_overrides"
            override_dir.mkdir(parents=True, exist_ok=True)

            if Path(obj_file_path).exists():
                target_path = override_dir / "triton_kernels_combined.o"
                shutil.copy(obj_file_path, target_path)
                log.info(f"Embedded combined kernel object: {target_path}")

                log.info(f"Successfully embedded {len(cubin_tuples)} kernel object files into single combined object")
                return [str(target_path)]
            else:
                log.warning(f"Combined object file not found: {obj_file_path}")
                return []

    except Exception as e:
        log.warning(f"Failed to embed CUBIN files: {e}")
        return []


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

        # NEW: Discover Triton kernels in overrides
        triton_kernels = discover_triton_kernels_in_overrides(override_graphs)
        log.info(f"Discovered {len(triton_kernels)} Triton kernels")

        # NEW: Compile Triton kernels using PyTorch Inductor infrastructure
        compiled_kernels = {}
        if triton_kernels:
            compiled_kernels = _compile_triton_kernels(triton_kernels)

        # NEW: Embed CUBIN binaries for compiled kernels
        embedded_objects = embed_override_cubins(output_dir, compiled_kernels)

        # NEW: Create runtime bridge for C++ kernel execution
        runtime_bridge_success = _create_runtime_bridge(compiled_kernels)
        if runtime_bridge_success:
            log.info("Created runtime bridge for C++ kernel execution")

        # Enhanced: Compile overrides to C++ with kernel information
        compiled_libs = compile_overrides(override_graphs, output_dir, compiled_kernels)

        # NEW: Setup runtime library loading
        library_setup_success = _setup_runtime_library_loading(output_dir, compiled_libs)
        if library_setup_success:
            log.info("Runtime library loading setup completed")

        compilation_time = time.time() - start_time

        if compiled_libs:
            log.info(f"Successfully compiled {len(compiled_libs)} override libraries in {compilation_time:.2f}s")
            result = {
                "status": "success",
                "compiled_libraries": compiled_libs,
                "generated_files": list(compiled_libs.values()),
                "statistics": {
                    "override_count": sum(len(ol) for ol in override_graphs.values()),
                    "library_count": len(compiled_libs),
                    "compilation_time": compilation_time,
                    "triton_kernels_discovered": len(triton_kernels),
                    "triton_kernels_compiled": len(compiled_kernels),
                    "cubin_objects_embedded": len(embedded_objects)
                }
            }
            return result
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

    # Add embedded kernel object files
    for obj_file in override_dir.glob("triton_kernel_*.o"):
        source_files.append(str(obj_file))

    # Add compiled override libraries (.so files)
    for so_file in override_dir.glob("*.so"):
        source_files.append(str(so_file))

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