"""
ReLU Operation Overrides

This module provides optimized ReLU implementations using various DSLs.
Currently supports:
- CuTeDSL/CUTLASS implementation for GPU acceleration
- Triton implementation for various tensor sizes and dtypes
"""

# Import both DSL implementations to register all overrides
from . import cutedsl_impl  # noqa: F401
from . import triton_impl   # noqa: F401