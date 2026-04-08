"""
Condition extraction from Python native override functions for AOTI compilation.

This module provides AST-based analysis to extract dispatch conditions from
Python native override functions, enabling their compilation into C++ code
for AOTInductor python-less deployment.
"""

import ast
import functools
import inspect
import textwrap
from typing import Any, Dict, Optional


class ConditionExtractor(ast.NodeVisitor):
    """Extracts dispatch conditions from Python override functions."""

    def __init__(self):
        self.conditions = []

    def extract(self, func) -> Dict[str, Any]:
        """Extract conditions with robust source handling."""
        try:
            # Unwrap partial functions
            func = func.func if isinstance(func, functools.partial) else func

            if func is None:
                return {"type": "always_true"}

            # Get source with multiple fallbacks
            source = self._get_source_robust(func)
            if not source:
                return {"type": "always_true"}

            # Parse with proper error handling
            try:
                # Clean up indentation issues
                source = textwrap.dedent(source)
                tree = ast.parse(source)
            except SyntaxError as e:
                return {"type": "error", "error": f"Syntax error: {str(e)}"}

            # Visit and extract
            self.conditions.clear()
            self.visit(tree)

            return self._serialize_conditions()

        except (OSError, IOError) as e:
            # File system errors - recoverable
            return {"type": "error", "error": f"File access error: {str(e)}"}
        except SyntaxError as e:
            # Source parsing errors - recoverable
            return {"type": "error", "error": f"Syntax error: {str(e)}"}

    def _get_source_robust(self, func) -> str:
        """Get function source with multiple fallback strategies."""
        try:
            # Strategy 1: Direct inspect.getsource
            return inspect.getsource(func)
        except (OSError, TypeError):
            pass

        try:
            # Strategy 2: Get from function code object
            if hasattr(func, '__code__'):
                filename = func.__code__.co_filename
                lineno = func.__code__.co_firstlineno

                # Try to read the file directly
                with open(filename, 'r') as f:
                    lines = f.readlines()

                # Find function definition
                func_lines = []
                in_function = False
                indent_level = None

                for i, line in enumerate(lines[lineno-1:], lineno):
                    if not in_function and f"def {func.__name__}" in line:
                        in_function = True
                        indent_level = len(line) - len(line.lstrip())
                        func_lines.append(line)
                    elif in_function:
                        current_indent = len(line) - len(line.lstrip()) if line.strip() else indent_level + 4
                        if line.strip() and current_indent <= indent_level:
                            break  # End of function
                        func_lines.append(line)

                if func_lines:
                    return ''.join(func_lines)
        except:
            pass

        # Strategy 3: Create simple template
        return f"""def {getattr(func, '__name__', 'unknown_func')}(x):
    return True  # Always true fallback
"""

    def visit_If(self, node):
        """Extract from if statements."""
        cond = self._extract_node(node.test)
        if cond:
            self.conditions.append(cond)
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        """Extract boolean operations."""
        op_type = "and" if isinstance(node.op, ast.And) else "or"
        sub_conds = [self._extract_node(v) for v in node.values]
        sub_conds = [c for c in sub_conds if c]
        if sub_conds:
            self.conditions.append({"type": op_type, "conditions": sub_conds})
        self.generic_visit(node)

    def _extract_node(self, node) -> Optional[Dict[str, Any]]:
        """Extract condition from any AST node."""
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            return self._extract_comparison(node.left, node.ops[0], node.comparators[0])
        elif isinstance(node, ast.Attribute):
            return self._extract_attribute(node)
        elif isinstance(node, ast.Call):
            return self._extract_call(node)
        elif isinstance(node, ast.BoolOp):
            op_type = "and" if isinstance(node.op, ast.And) else "or"
            sub_conds = [self._extract_node(v) for v in node.values]
            sub_conds = [c for c in sub_conds if c]
            return {"type": op_type, "conditions": sub_conds} if sub_conds else None
        return None

    def _extract_comparison(self, left, op, right):
        """Extract comparison operations."""
        # Handle x.dtype == torch.float32
        if isinstance(left, ast.Attribute) and left.attr == "dtype" and isinstance(op, ast.Eq):
            param = self._get_name(left.value)
            dtype = self._get_dtype(right)
            return {"type": "dtype_eq", "param": param, "value": dtype} if param and dtype else None

        # Handle x.numel() >= 1024
        elif isinstance(left, ast.Call) and isinstance(left.func, ast.Attribute):
            if left.func.attr == "numel" and isinstance(op, (ast.Gt, ast.GtE)):
                param = self._get_name(left.func.value)
                value = self._get_number(right)
                return {"type": "numel_gte", "param": param, "value": value} if param and value is not None else None

            # Handle x.size(0) >= 32
            elif left.func.attr == "size" and len(left.args) == 1:
                param = self._get_name(left.func.value)
                dim = self._get_number(left.args[0])
                value = self._get_number(right)
                if param and dim is not None and value is not None:
                    if isinstance(op, (ast.Gt, ast.GtE)):
                        return {"type": "size_dim_gte", "param": param, "dim": dim, "value": value}
                    elif isinstance(op, (ast.Lt, ast.LtE)):
                        return {"type": "size_dim_lte", "param": param, "dim": dim, "value": value}
                    elif isinstance(op, ast.Eq):
                        return {"type": "size_dim_eq", "param": param, "dim": dim, "value": value}

        # Handle cross-dimensional comparisons: x.shape[0] == x.shape[1] (MUST be checked first!)
        elif (isinstance(left, ast.Subscript) and isinstance(right, ast.Subscript) and
              isinstance(left.value, ast.Attribute) and isinstance(right.value, ast.Attribute) and
              left.value.attr == "shape" and right.value.attr == "shape" and isinstance(op, ast.Eq)):
            param_left = self._get_name(left.value.value)
            param_right = self._get_name(right.value.value)
            dim_left = self._get_number(left.slice)
            dim_right = self._get_number(right.slice)

            if (param_left and param_right and dim_left is not None and dim_right is not None and
                param_left == param_right):  # Same tensor
                return {"type": "shape_dim_eq_cross", "param": param_left, "dim1": dim_left, "dim2": dim_right}

        # Handle x.shape[0] >= 100 (checked after cross-dimensional)
        elif isinstance(left, ast.Subscript) and isinstance(left.value, ast.Attribute) and left.value.attr == "shape":
            param = self._get_name(left.value.value)
            dim = self._get_number(left.slice)
            value = self._get_number(right)
            if param and dim is not None and value is not None:
                if isinstance(op, (ast.Gt, ast.GtE)):
                    return {"type": "shape_dim_gte", "param": param, "dim": dim, "value": value}
                elif isinstance(op, (ast.Lt, ast.LtE)):
                    return {"type": "shape_dim_lte", "param": param, "dim": dim, "value": value}
                elif isinstance(op, ast.Eq):
                    return {"type": "shape_dim_eq", "param": param, "dim": dim, "value": value}

        # Handle x.ndim == 2
        elif isinstance(left, ast.Attribute) and left.attr == "ndim":
            param = self._get_name(left.value)
            value = self._get_number(right)
            if param and value is not None:
                if isinstance(op, ast.Eq):
                    return {"type": "ndim_eq", "param": param, "value": value}
                elif isinstance(op, (ast.Gt, ast.GtE)):
                    return {"type": "ndim_gte", "param": param, "value": value}
                elif isinstance(op, (ast.Lt, ast.LtE)):
                    return {"type": "ndim_lte", "param": param, "value": value}

        return None

    def _extract_attribute(self, node):
        """Extract attribute checks like x.is_cuda."""
        if node.attr == "is_cuda":
            param = self._get_name(node.value)
            return {"type": "is_cuda", "param": param} if param else None
        elif node.attr == "ndim":
            # Handle x.ndim when used directly in boolean context
            param = self._get_name(node.value)
            return {"type": "ndim_truthy", "param": param} if param else None
        return None

    def _extract_call(self, node):
        """Extract method calls."""
        if isinstance(node.func, ast.Attribute) and node.func.attr == "is_cuda":
            param = self._get_name(node.func.value)
            return {"type": "is_cuda", "param": param} if param else None
        return None

    def _get_name(self, node):
        """Get parameter name."""
        return node.id if isinstance(node, ast.Name) else None

    def _get_dtype(self, node):
        """Get dtype value."""
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "torch":
            return f"torch.{node.attr}"
        return None

    def _get_number(self, node):
        """Get numeric value with Python version compatibility."""
        # Handle ast.Constant (Python 3.8+)
        if hasattr(ast, 'Constant') and isinstance(node, ast.Constant):
            value = getattr(node, 'value', None)
            if isinstance(value, (int, float)):
                return int(value)
        # Handle ast.Num (Python < 3.8)
        elif hasattr(ast, 'Num') and isinstance(node, ast.Num):
            value = getattr(node, 'n', None)
            if isinstance(value, (int, float)):
                return int(value)
        # Handle expressions like 16*1024*1024 using safe evaluation
        elif isinstance(node, ast.BinOp):
            try:
                # Only evaluate safe numeric expressions
                if self._is_safe_numeric(node):
                    return self._safe_evaluate_numeric(node)
            except (ValueError, TypeError):
                pass
        return None

    def _is_safe_numeric(self, node):
        """Check if node represents a safe numeric expression."""
        if hasattr(ast, 'Constant') and isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float))
        elif hasattr(ast, 'Num') and isinstance(node, ast.Num):
            return isinstance(node.n, (int, float))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub)):
            return self._is_safe_numeric(node.left) and self._is_safe_numeric(node.right)
        return False

    def _safe_evaluate_numeric(self, node):
        """Safely evaluate numeric expressions without eval()."""
        if hasattr(ast, 'Constant') and isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return int(node.value)
        elif hasattr(ast, 'Num') and isinstance(node, ast.Num):
            if isinstance(node.n, (int, float)):
                return int(node.n)
        elif isinstance(node, ast.BinOp):
            left = self._safe_evaluate_numeric(node.left)
            right = self._safe_evaluate_numeric(node.right)
            if left is not None and right is not None:
                if isinstance(node.op, ast.Mult):
                    return left * right
                elif isinstance(node.op, ast.Add):
                    return left + right
                elif isinstance(node.op, ast.Sub):
                    return left - right
        raise ValueError(f"Cannot safely evaluate node type: {type(node)}")

    def _serialize_conditions(self):
        """Serialize conditions."""
        if not self.conditions:
            return {"type": "always_true"}
        elif len(self.conditions) == 1:
            return self.conditions[0]
        else:
            return {"type": "and", "conditions": self.conditions}


# Single function API
def extract_conditions(func) -> Dict[str, Any]:
    """Extract dispatch conditions from override function."""
    return ConditionExtractor().extract(func)


def extract_conditions_from_source(source_code: str) -> Dict[str, Any]:
    """Extract dispatch conditions directly from source code string."""
    extractor = ConditionExtractor()
    try:
        # Clean up indentation issues
        source = textwrap.dedent(source_code).strip()
        tree = ast.parse(source)

        # Visit and extract
        extractor.conditions.clear()
        extractor.visit(tree)

        return extractor._serialize_conditions()

    except SyntaxError as e:
        return {"type": "error", "error": f"Syntax error: {str(e)}"}
    except Exception as e:
        return {"type": "error", "error": f"Extraction error: {str(e)}"}