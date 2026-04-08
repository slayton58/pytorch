"""
C++ code generation for Python native override compilation in AOTI.

This module generates C++ implementation files from Python native override
dispatch conditions, enabling their integration into AOTInductor compiled
models for python-less deployment environments.
"""

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from torch._inductor.utils import IndentedBuffer
from torch._inductor.cpp_builder import CppBuilder, CppTorchOptions
from torch._inductor.codecache import CppCodeCache
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.codegen.cpp_utils import DTYPE_TO_ATEN

from .python_native_aoti_condition_extraction import extract_conditions


class CodeGenerator:
    """Generates C++ code for PyTorch override compilation."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("/tmp/aoti_overrides")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Use PyTorch's existing systems with correct APIs
        self.cpp_cache = CppCodeCache()
        self.async_compile = AsyncCompile()

        # Simple in-memory cache for this session
        self._compilation_cache = {}
        self._last_compilation_errors = []

    def compile_override_graphs(self, override_graphs: Dict) -> Dict[str, str]:
        """Compile all override graphs with caching and improved error tracking."""
        results = {}
        compilation_errors = []

        for (op_name, dispatch_key), override_list in override_graphs.items():
            try:
                # Generate cache key
                cache_key = self._cache_key(op_name, dispatch_key, override_list)

                # Check simple cache first
                if cache_key in self._compilation_cache:
                    results[f"{op_name}_{dispatch_key}"] = self._compilation_cache[cache_key]
                    continue

                # Compile and cache
                lib_path = self._compile_single_graph(op_name, dispatch_key, override_list)
                if lib_path:
                    self._compilation_cache[cache_key] = lib_path
                    results[f"{op_name}_{dispatch_key}"] = lib_path
                else:
                    compilation_errors.append(f"{op_name}_{dispatch_key}: Compilation returned no output")

            except Exception as e:
                error_msg = f"Failed to compile {op_name}_{dispatch_key}: {e}"
                print(error_msg)
                compilation_errors.append(error_msg)

        # Store compilation errors for later access
        self._last_compilation_errors = compilation_errors
        return results

    def _compile_single_graph(self, op_name: str, dispatch_key: str, override_list: List) -> str:
        """Compile single override graph."""
        # Extract and serialize conditions
        override_specs = []
        for i, override_node in enumerate(override_list):
            conditions = extract_conditions(override_node.override_fn)
            spec = {
                "name": f"{op_name}_{dispatch_key}_{i}",
                "op_name": op_name,
                "dispatch_key": dispatch_key,
                "conditions": conditions,
                "dsl_name": getattr(override_node, 'dsl_name', 'unknown')
            }
            override_specs.append(spec)

        # Generate C++ using PyTorch's IndentedBuffer
        cpp_content = self._generate_cpp(override_specs)

        # Write and compile
        cpp_file = self.cache_dir / f"{op_name}_{dispatch_key}.cpp"
        cpp_file.write_text(cpp_content)

        return self._compile_cpp(str(cpp_file), f"{op_name}_{dispatch_key}")

    def _generate_cpp(self, override_specs: List[Dict]) -> str:
        """Generate C++ using PyTorch's IndentedBuffer."""
        buffer = IndentedBuffer()

        # Headers
        buffer.writeline("#include <torch/library.h>")
        buffer.writeline("#include <ATen/ATen.h>")
        buffer.writeline("#include <ATen/core/dispatch/Dispatcher.h>")
        buffer.writeline("#include <ATen/core/boxing/KernelFunction.h>")
        buffer.writeline("#include <c10/core/DispatchKey.h>")
        buffer.writeline("#include <c10/core/DispatchKeySet.h>")
        buffer.writeline("#include <torch/csrc/jit/runtime/operator.h>")
        buffer.writeline("#include <torch/csrc/jit/runtime/jit_exception.h>")
        buffer.writeline("#include <mutex>")
        buffer.writeline("#include <optional>")
        buffer.newline()

        # Generate helper functions in named namespace
        buffer.writeline("namespace aoti_overrides {")
        with buffer.indent():
            # Generate condition checkers
            for spec in override_specs:
                self._write_condition_checker(buffer, spec)
                buffer.newline()

            # Generate dispatchers
            for spec in override_specs:
                self._write_dispatcher(buffer, spec)
                buffer.newline()

        buffer.writeline("} // namespace aoti_overrides")
        buffer.newline()

        # Follow TorchGen pattern: TORCH_LIBRARY_IMPL in anonymous namespace
        buffer.writeline("// TORCH_LIBRARY_IMPL must be in anonymous namespace (TorchGen convention)")
        buffer.writeline("namespace {")
        with buffer.indent():
            registrations = self._group_by_op(override_specs)
            for (op_name, dispatch_key), specs in registrations.items():
                self._write_registration(buffer, op_name, dispatch_key, specs)
                buffer.newline()

        buffer.writeline("} // anonymous namespace")
        return buffer.getvalue()

    def _write_condition_checker(self, buffer: IndentedBuffer, spec: Dict):
        """Write condition checker function."""
        name = spec["name"]
        conditions = spec["conditions"]

        buffer.writeline(f"bool check_{name}_conditions(const at::Tensor& x) {{")
        with buffer.indent():
            expr = self._convert_conditions(conditions)
            buffer.writeline(f"return {expr};")
        buffer.writeline("}")

    def _write_dispatcher(self, buffer: IndentedBuffer, spec: Dict):
        """
        Write dispatch function using proper kernel preservation pattern.

        This uses the C++ equivalent of torch.library.get_kernel to access
        the fallback implementation properly.
        """
        name = spec["name"]
        op_name = spec["op_name"]
        dispatch_key = spec["dispatch_key"]

        # Get current implementation before override (C++ equivalent of torch.library.get_kernel)
        full_op_name = f"aten::{op_name}"
        buffer.writeline(f"// Get current implementation before registering override")
        buffer.writeline(f"c10::SafeKernelFunction get_current_{name}_kernel() {{")
        with buffer.indent():
            buffer.writeline(f'auto op = c10::Dispatcher::singleton().findOp(torch::jit::parseName("{full_op_name}"));')
            buffer.writeline("TORCH_CHECK(op.has_value(), \"Operator not found\");")
            buffer.writeline(f"return op->getComputedKernelForDispatchKey(c10::DispatchKey::{dispatch_key});")
        buffer.writeline("}")
        buffer.newline()

        buffer.writeline(f"// Store the current implementation (before our override)")
        buffer.writeline(f"static c10::SafeKernelFunction {name}_fallback_kernel = get_current_{name}_kernel();")
        buffer.newline()

        buffer.writeline(f"// Fallback using stored implementation")
        buffer.writeline(f"at::Tensor fallback_{name}(const at::Tensor& x) {{")
        with buffer.indent():
            buffer.writeline("torch::jit::Stack stack;")
            buffer.writeline("stack.push_back(x);")
            buffer.writeline(f'auto op = c10::Dispatcher::singleton().findOp(torch::jit::parseName("{full_op_name}"));')
            buffer.writeline(f"{name}_fallback_kernel.callBoxed(*op, c10::DispatchKeySet(), &stack);")
            buffer.writeline("return stack[0].toTensor();")
        buffer.writeline("}")
        buffer.newline()

        # Detect if this is an inplace operation (ends with _ or contains inplace indicator)
        is_inplace = op_name.endswith('_') or (op_name.count('.') > 0 and op_name.split('.')[-1].endswith('_'))

        # Generate the actual dispatcher function with correct signature
        buffer.writeline(f"// Override dispatcher for {op_name}")

        if is_inplace:
            # Inplace operations: take non-const reference, return reference
            buffer.writeline(f"at::Tensor& {name}_dispatch(at::Tensor& x) {{")
        else:
            # Non-inplace operations: take const reference, return by value
            buffer.writeline(f"at::Tensor {name}_dispatch(const at::Tensor& x) {{")

        with buffer.indent():
            buffer.writeline(f"if (check_{name}_conditions(x)) {{")
            with buffer.indent():
                # For now, just call fallback even when conditions match
                # TODO: Call actual optimized kernel here
                buffer.writeline(f"// TODO: Call optimized {spec['dsl_name']} kernel")
                buffer.writeline(f"// For now, use fallback even when conditions match")
                if is_inplace:
                    buffer.writeline(f"at::Tensor result = fallback_{name}(x);")
                    buffer.writeline("x.copy_(result);")
                    buffer.writeline("return x;")
                else:
                    buffer.writeline(f"return fallback_{name}(x);")
            buffer.writeline("}")

            # Use fallback kernel
            buffer.writeline("// Call fallback implementation")
            if is_inplace:
                # For inplace operations
                buffer.writeline(f"at::Tensor result = fallback_{name}(x);")
                buffer.writeline("x.copy_(result);  // Copy result back to input tensor for inplace semantics")
                buffer.writeline("return x;")
            else:
                # For non-inplace operations
                buffer.writeline(f"return fallback_{name}(x);")
        buffer.writeline("}")

    def _write_registration(self, buffer: IndentedBuffer, op_name: str, dispatch_key: str, specs: List):
        """Write TORCH_LIBRARY_IMPL registration with proper inplace/non-inplace signatures."""
        buffer.writeline(f"TORCH_LIBRARY_IMPL(aten, {dispatch_key}, m) {{")
        with buffer.indent():
            for spec in specs:
                # Register the dispatcher with correct signature (inplace vs non-inplace)
                buffer.writeline(f'm.impl("{op_name}", aoti_overrides::{spec["name"]}_dispatch);')
        buffer.writeline("}")

    def _convert_conditions(self, conditions: Dict) -> str:
        """Convert condition dict to C++ expression."""
        cond_type = conditions.get("type", "always_true")

        if cond_type == "always_true":
            return "true"
        elif cond_type == "error":
            return "true"  # Safe fallback
        elif cond_type == "and":
            sub_exprs = [self._convert_conditions(c) for c in conditions.get("conditions", [])]
            sub_exprs = [e for e in sub_exprs if e != "true"]  # Filter out trivial conditions
            return f"({' && '.join(sub_exprs)})" if sub_exprs else "true"
        elif cond_type == "or":
            sub_exprs = [self._convert_conditions(c) for c in conditions.get("conditions", [])]
            return f"({' || '.join(sub_exprs)})" if sub_exprs else "true"
        elif cond_type == "dtype_eq":
            param = conditions.get("param", "x")
            dtype_str = conditions.get("value", "torch.float32")

            # Convert string to torch dtype and use standard mapping
            dtype_map = {
                "torch.float32": torch.float32,
                "torch.bfloat16": torch.bfloat16,
                "torch.float16": torch.float16,
                "torch.int32": torch.int32,
                "torch.int64": torch.int64
            }
            torch_dtype = dtype_map.get(dtype_str, torch.float32)
            aten_dtype = DTYPE_TO_ATEN.get(torch_dtype, "at::kFloat")
            return f"{param}.dtype() == {aten_dtype}"
        elif cond_type == "numel_gte":
            param = conditions.get("param", "x")
            value = conditions.get("value", 0)
            return f"{param}.numel() >= {value}L"
        elif cond_type == "is_cuda":
            param = conditions.get("param", "x")
            return f"{param}.is_cuda()"
        elif cond_type == "shape_dim_gte":
            param = conditions.get("param", "x")
            dim = conditions.get("dim", 0)
            value = conditions.get("value", 0)
            return f"{param}.size({dim}) >= {value}"
        elif cond_type == "shape_dim_lte":
            param = conditions.get("param", "x")
            dim = conditions.get("dim", 0)
            value = conditions.get("value", 0)
            return f"{param}.size({dim}) <= {value}"
        elif cond_type == "shape_dim_eq":
            param = conditions.get("param", "x")
            dim = conditions.get("dim", 0)
            value = conditions.get("value", 0)
            return f"{param}.size({dim}) == {value}"
        elif cond_type == "size_dim_gte":
            param = conditions.get("param", "x")
            dim = conditions.get("dim", 0)
            value = conditions.get("value", 0)
            return f"{param}.size({dim}) >= {value}"
        elif cond_type == "size_dim_lte":
            param = conditions.get("param", "x")
            dim = conditions.get("dim", 0)
            value = conditions.get("value", 0)
            return f"{param}.size({dim}) <= {value}"
        elif cond_type == "size_dim_eq":
            param = conditions.get("param", "x")
            dim = conditions.get("dim", 0)
            value = conditions.get("value", 0)
            return f"{param}.size({dim}) == {value}"
        elif cond_type == "ndim_eq":
            param = conditions.get("param", "x")
            value = conditions.get("value", 2)
            return f"{param}.ndimension() == {value}"
        elif cond_type == "ndim_gte":
            param = conditions.get("param", "x")
            value = conditions.get("value", 2)
            return f"{param}.ndimension() >= {value}"
        elif cond_type == "ndim_lte":
            param = conditions.get("param", "x")
            value = conditions.get("value", 2)
            return f"{param}.ndimension() <= {value}"
        elif cond_type == "ndim_truthy":
            param = conditions.get("param", "x")
            return f"{param}.ndimension() > 0"
        elif cond_type == "shape_dim_eq_cross":
            param = conditions.get("param", "x")
            dim1 = conditions.get("dim1", 0)
            dim2 = conditions.get("dim2", 1)
            return f"{param}.size({dim1}) == {param}.size({dim2})"
        else:
            return "true"  # Safe fallback

    # Removed _generate_fallback_call - now using proper kernel preservation pattern

    def _group_by_op(self, specs: List[Dict]) -> Dict:
        """Group specs by (op_name, dispatch_key)."""
        groups = {}
        for spec in specs:
            key = (spec["op_name"], spec["dispatch_key"])
            if key not in groups:
                groups[key] = []
            groups[key].append(spec)
        return groups

    def _compile_cpp(self, cpp_file: str, name: str) -> str:
        """Compile C++ file using PyTorch's standard CppBuilder infrastructure."""
        try:
            # Use PyTorch's CppTorchOptions which includes PyTorch headers and libraries

            # Use torch-aware options with appropriate flags for override compilation
            build_options = CppTorchOptions(
                include_pytorch=True,    # Include PyTorch headers (torch/library.h, ATen/ATen.h)
                compile_only=False,      # We want a shared library
                warning_all=False,       # Reduce noise in compilation
                use_relative_path=False, # Use absolute paths for reliability
                aot_mode=True,          # We're in AOT compilation mode
                shared=True,            # Generate shared library
                extra_flags=["-fPIC", "-O2"]  # Position-independent code + optimization
            )

            builder = CppBuilder(
                name=name,
                sources=[cpp_file],
                BuildOption=build_options,
                output_dir=str(self.cache_dir)
            )

            # CppBuilder.build() returns None even on success, so we need to check for the output file
            builder.build()

            # Construct expected library path
            lib_path = self.cache_dir / f"{name}.so"

            if lib_path.exists():
                return str(lib_path)
            else:
                # If .so doesn't exist, try other common extensions
                for ext in ['.dylib', '.dll']:
                    alt_path = self.cache_dir / f"{name}{ext}"
                    if alt_path.exists():
                        return str(alt_path)

                raise RuntimeError(f"Compilation succeeded but output library not found: expected {lib_path}")

        except FileNotFoundError as e:
            # C++ compiler not found - clear error message
            raise RuntimeError(f"C++ compiler not found. Please install g++ or configure build environment: {e}")
        except Exception as e:
            # Compilation failed - fail properly with details
            raise RuntimeError(f"C++ compilation failed for {name}: {e}")

    def _cache_key(self, op_name: str, dispatch_key: str, override_list: List) -> str:
        """Generate cache key."""
        content = f"{op_name}_{dispatch_key}_{len(override_list)}"
        for override_node in override_list:
            content += f"_{getattr(override_node, 'dsl_name', 'unknown')}"
            # Use function name instead of full function object for stability
            func_name = getattr(override_node.override_fn, '__name__', 'anonymous')
            content += f"_{func_name}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_last_compilation_errors(self) -> List[str]:
        """Get compilation errors from the last compilation run."""
        return self._last_compilation_errors.copy()

    def has_compilation_errors(self) -> bool:
        """Check if the last compilation had any errors."""
        return len(self._last_compilation_errors) > 0


# Single function API
def compile_overrides(override_graphs: Dict, cache_dir: Optional[str] = None) -> Dict[str, str]:
    """Compile override graphs to C++ libraries."""
    generator = CodeGenerator(cache_dir)
    return generator.compile_override_graphs(override_graphs)