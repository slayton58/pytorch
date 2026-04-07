"""
C++ code generation for PyTorch override compilation.
"""

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch._inductor.utils import IndentedBuffer
from torch._inductor.cpp_builder import CppBuilder
from torch._inductor.codecache import CppCodeCache
from torch._inductor.async_compile import AsyncCompile

from .condition_extraction import extract_conditions


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

    def compile_override_graphs(self, override_graphs: Dict) -> Dict[str, str]:
        """Compile all override graphs with caching."""
        results = {}

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

            except Exception as e:
                print(f"Failed to compile {op_name}_{dispatch_key}: {e}")

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
        buffer.newline()

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

            # Generate registrations
            registrations = self._group_by_op(override_specs)
            for (op_name, dispatch_key), specs in registrations.items():
                self._write_registration(buffer, op_name, dispatch_key, specs)
                buffer.newline()

        buffer.writeline("}")
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
        """Write dispatch function."""
        name = spec["name"]
        op_name = spec["op_name"]

        buffer.writeline(f"at::Tensor {name}_dispatch(const at::Tensor& x) {{")
        with buffer.indent():
            buffer.writeline(f"if (check_{name}_conditions(x)) {{")
            with buffer.indent():
                # Placeholder for actual kernel call
                buffer.writeline(f"// Call optimized {spec['dsl_name']} kernel")
                buffer.writeline("// return optimized_kernel(x);")
            buffer.writeline("}")
            buffer.writeline(f"return at::native::{op_name}(x);")
        buffer.writeline("}")

    def _write_registration(self, buffer: IndentedBuffer, op_name: str, dispatch_key: str, specs: List):
        """Write TORCH_LIBRARY_IMPL registration."""
        buffer.writeline(f"TORCH_LIBRARY_IMPL(aten, {dispatch_key}, m) {{")
        with buffer.indent():
            for spec in specs:
                buffer.writeline(f'm.impl("{op_name}", {spec["name"]}_dispatch);')
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
            dtype_map = {
                "torch.float32": "at::kFloat",
                "torch.bfloat16": "at::kBFloat16",
                "torch.float16": "at::kHalf",
                "torch.int32": "at::kInt",
                "torch.int64": "at::kLong"
            }
            param = conditions.get("param", "x")
            dtype = dtype_map.get(conditions.get("value"), "at::kFloat")
            return f"{param}.dtype() == {dtype}"
        elif cond_type == "numel_gte":
            param = conditions.get("param", "x")
            value = conditions.get("value", 0)
            return f"{param}.numel() >= {value}L"
        elif cond_type == "is_cuda":
            param = conditions.get("param", "x")
            return f"{param}.is_cuda()"
        else:
            return "true"  # Safe fallback

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
        """Compile C++ file using PyTorch's CppBuilder."""
        try:
            # Use simplified build options
            from torch._inductor.cpp_builder import get_cpp_compiler, BuildOptionsBase

            class SimpleBuildOptions(BuildOptionsBase):
                def get_compiler(self):
                    return get_cpp_compiler()

                def get_use_relative_path(self):
                    return False

                def get_aot_mode(self):
                    return True

                def get_compile_only(self):
                    return False

                def get_precompiling(self):
                    return False

                def get_preprocessing(self):
                    return False

            builder = CppBuilder(
                name=name,
                sources=[cpp_file],
                BuildOption=SimpleBuildOptions(),
                output_dir=str(self.cache_dir)
            )

            return builder.build()

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


# Single function API
def compile_overrides(override_graphs: Dict, cache_dir: Optional[str] = None) -> Dict[str, str]:
    """Compile override graphs to C++ libraries."""
    generator = CodeGenerator(cache_dir)
    return generator.compile_override_graphs(override_graphs)