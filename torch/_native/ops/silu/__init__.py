"""
Triton SiLU native op implementation.

Provides conditional override of aten::silu for large bfloat16 tensors on CUDA.
"""

from . import triton_impl