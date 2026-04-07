"""
Triton ReLU Implementation

This module provides Triton-based ReLU overrides with different conditions
than the CuTeDSL ones, demonstrating multi-DSL override coexistence.
"""

import functools
import torch
from ... import triton_utils as tu


def triton_relu_large_dispatch(dispatch_keys: torch.DispatchKeySet, x: torch.Tensor, *, fallback_kernel):
    """
    Triton ReLU dispatch optimized for very large tensors.

    Conditions (different from CuTeDSL):
    - Use Triton for very large tensors (>= 1M elements)
    - Any floating point dtype (broader than CuTeDSL)
    - CUDA device required
    - Optimized for memory bandwidth on large tensors
    """
    # Lazy import to avoid loading Triton at registration time
    from .triton_kernels import triton_relu_large_kernel_launcher

    # Triton conditions - optimized for very large tensors
    use_triton = (
        x.dtype.is_floating_point and           # Any float dtype (fp16, fp32, bf16)
        x.numel() >= 1024 * 1024 and            # 1M+ elements (larger than CuTeDSL threshold)
        x.is_cuda and                           # CUDA device
        x.is_contiguous()                       # Memory layout requirement
    )

    if use_triton:
        return triton_relu_large_kernel_launcher(x)
    else:
        return fallback_kernel.call_boxed(dispatch_keys, x)


def triton_relu_fp32_dispatch(dispatch_keys: torch.DispatchKeySet, x: torch.Tensor, *, fallback_kernel):
    """
    Triton ReLU dispatch specialized for float32 tensors.

    Conditions (complementary to CuTeDSL):
    - Specifically optimized for float32 (CuTeDSL focuses on fp16/bf16)
    - Medium-sized tensors (different range than other overrides)
    - Uses Triton's float32 optimizations
    """
    from .triton_kernels import triton_relu_fp32_kernel_launcher

    # Triton float32 specialization
    use_triton_fp32 = (
        x.dtype == torch.float32 and            # Specific to float32
        x.numel() >= 16384 and                  # 16K+ elements (between CuTeDSL thresholds)
        x.numel() < 1024 * 1024 and             # < 1M elements (below large tensor override)
        x.is_cuda and
        x.is_contiguous()
    )

    if use_triton_fp32:
        return triton_relu_fp32_kernel_launcher(x)
    else:
        return fallback_kernel.call_boxed(dispatch_keys, x)


def triton_relu_inplace_dispatch(dispatch_keys: torch.DispatchKeySet, x: torch.Tensor, *, fallback_kernel):
    """
    Triton in-place ReLU dispatch with different conditions than CuTeDSL.

    Conditions:
    - Optimized for double precision and large float32 tensors
    - Different size threshold than CuTeDSL in-place
    """
    from .triton_kernels import triton_relu_inplace_kernel_launcher

    # Triton in-place specialization
    use_triton_inplace = (
        (x.dtype == torch.float64 or
         (x.dtype == torch.float32 and x.numel() >= 32768)) and  # Different conditions
        x.is_cuda and
        x.is_contiguous()
    )

    if use_triton_inplace:
        # For in-place, modify the tensor directly
        x.copy_(triton_relu_inplace_kernel_launcher(x))
        return x
    else:
        return fallback_kernel.call_boxed(dispatch_keys, x)


def register_triton_relu_overrides():
    """
    Register multiple Triton ReLU overrides with different conditions.

    These will coexist with CuTeDSL overrides, demonstrating multi-DSL
    override handling and priority resolution.
    """
    # Get fallback kernels
    relu_fallback = torch.library.get_kernel("aten::relu", "CUDA")
    relu_inplace_fallback = torch.library.get_kernel("aten::relu_", "CUDA")

    # Create dispatch functions
    relu_large_fn = functools.partial(
        triton_relu_large_dispatch,
        fallback_kernel=relu_fallback
    )

    relu_fp32_fn = functools.partial(
        triton_relu_fp32_dispatch,
        fallback_kernel=relu_fallback
    )

    relu_inplace_fn = functools.partial(
        triton_relu_inplace_dispatch,
        fallback_kernel=relu_inplace_fallback
    )

    # Register with Triton utils
    # Note: Registration order matters - last registered is first checked

    # Register in order of specificity (most specific first)
    tu.register_op_override(
        "aten",
        "relu",
        "CUDA",
        relu_fp32_fn  # Most specific: float32 medium-sized tensors
    )

    tu.register_op_override(
        "aten",
        "relu",
        "CUDA",
        relu_large_fn  # Less specific: very large tensors any float dtype
    )

    tu.register_op_override(
        "aten",
        "relu_",
        "CUDA",
        relu_inplace_fn  # In-place specialization
    )


# Register Triton ReLU overrides when this module is imported
register_triton_relu_overrides()