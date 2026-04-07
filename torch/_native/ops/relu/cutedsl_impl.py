"""
CuTeDSL ReLU Implementation

This module provides conditional dispatch logic for CuTeDSL ReLU kernels,
demonstrating the same override pattern as SiLU but for a different DSL.
"""

import functools
import torch
from ... import cutedsl_utils as cu


def cutedsl_relu_dispatch(dispatch_keys: torch.DispatchKeySet, x: torch.Tensor, *, fallback_kernel):
    """
    Conditional dispatch function for CuTeDSL ReLU.

    This function implements the decision logic for when to use the optimized
    CuTeDSL ReLU kernel vs falling back to the standard implementation.

    Args:
        dispatch_keys: PyTorch dispatch key set
        x: Input tensor
        fallback_kernel: Standard ReLU implementation to fall back to

    Returns:
        ReLU result from either CuTeDSL kernel or fallback

    Condition Logic (extractable by our system):
        - Use CuTeDSL when:
          * tensor dtype is float16 or bfloat16 (optimized for these types)
          * tensor has at least 4096 elements (overhead threshold)
          * tensor is on CUDA device
          * tensor is contiguous (required for CuTeDSL efficiency)
    """
    # Lazy import to avoid loading CuTeDSL at registration time
    from .cutedsl_kernels import cutedsl_relu_kernel_launcher

    # Conditional logic - this is what our system will extract and convert to C++
    use_cutedsl = (
        (x.dtype == torch.float16 or x.dtype == torch.bfloat16) and  # Optimized dtypes
        x.numel() >= 4096 and                                        # Size threshold
        x.is_cuda and                                                # CUDA device
        x.is_contiguous()                                           # Memory layout
    )

    if use_cutedsl:
        return cutedsl_relu_kernel_launcher(x)
    else:
        return fallback_kernel.call_boxed(dispatch_keys, x)


def cutedsl_relu_inplace_dispatch(dispatch_keys: torch.DispatchKeySet, x: torch.Tensor, *, fallback_kernel):
    """
    Conditional dispatch for in-place CuTeDSL ReLU.

    Similar to the regular version but for in-place operations.
    Uses slightly different conditions to demonstrate variety.
    """
    from .cutedsl_kernels import cutedsl_relu_kernel_launcher

    # Different conditions for in-place to show variety in our system
    use_cutedsl_inplace = (
        x.dtype in [torch.float16, torch.bfloat16, torch.float32] and  # Broader dtype support
        x.numel() >= 8192 and                                          # Higher threshold for inplace
        x.is_cuda and
        x.is_contiguous()
    )

    if use_cutedsl_inplace:
        # For in-place, modify the tensor directly
        x.copy_(cutedsl_relu_kernel_launcher(x))
        return x
    else:
        return fallback_kernel.call_boxed(dispatch_keys, x)


def cutedsl_relu_small_dispatch(dispatch_keys: torch.DispatchKeySet, x: torch.Tensor, *, fallback_kernel):
    """
    CuTeDSL dispatch optimized for small, specific tensors.

    Conditions (complementary to other CuTeDSL override):
    - Small tensors that benefit from CuTeDSL's template specialization
    - Specific dtype combinations
    - Different from the large tensor CuTeDSL override
    """
    from .cutedsl_kernels import cutedsl_relu_small_kernel_launcher

    # Small tensor CuTeDSL specialization
    use_cutedsl_small = (
        x.dtype == torch.bfloat16 and           # Specific to bfloat16
        x.numel() >= 1024 and                   # Smaller threshold
        x.numel() < 4096 and                    # Below main CuTeDSL threshold
        x.is_cuda and
        x.is_contiguous()
    )

    if use_cutedsl_small:
        return cutedsl_relu_small_kernel_launcher(x)
    else:
        return fallback_kernel.call_boxed(dispatch_keys, x)


def register_cutedsl_overrides():
    """
    Register multiple CuTeDSL ReLU overrides with the dispatcher.

    This follows the same pattern as SiLU registration but for ReLU ops.
    Now includes multiple CuTeDSL variants for different scenarios.
    """
    # Get fallback kernels for conditional dispatch
    relu_fallback = torch.library.get_kernel("aten::relu", "CUDA")
    relu_inplace_fallback = torch.library.get_kernel("aten::relu_", "CUDA")

    # Create partially applied dispatch functions
    relu_dispatch_fn = functools.partial(
        cutedsl_relu_dispatch,
        fallback_kernel=relu_fallback
    )

    relu_small_dispatch_fn = functools.partial(
        cutedsl_relu_small_dispatch,
        fallback_kernel=relu_fallback
    )

    relu_inplace_dispatch_fn = functools.partial(
        cutedsl_relu_inplace_dispatch,
        fallback_kernel=relu_inplace_fallback
    )

    # Register with the CuTeDSL utils
    # Registration order matters - last registered is checked first

    cu.register_op_override(
        "aten",
        "relu",
        "CUDA",
        relu_small_dispatch_fn  # Small tensor specialization
    )

    cu.register_op_override(
        "aten",
        "relu",
        "CUDA",
        relu_dispatch_fn  # Main CuTeDSL override
    )

    cu.register_op_override(
        "aten",
        "relu_",
        "CUDA",
        relu_inplace_dispatch_fn
    )


# Register overrides when this module is imported
register_cutedsl_overrides()