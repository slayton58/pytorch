"""
Triton SiLU implementation registration for torch._native.

Provides conditional override of aten::silu and aten::silu_ for CUDA with specific conditions:
- Input dtype must be bfloat16
- Input must have at least 16M elements (16*1024*1024)
"""

import torch

from ... import triton_utils as tu


def triton_silu_dispatch(
    dispatch_keys: torch.DispatchKeySet,
    x: torch.Tensor,
    *,
    fallback_kernel,
) -> torch.Tensor:
    """
    Conditional dispatch for Triton SiLU implementation.

    Uses Triton implementation when:
    - Input dtype is bfloat16 AND
    - Input has at least 16M elements

    Otherwise falls back to PyTorch's native implementation.

    Args:
        dispatch_keys: PyTorch dispatch keys
        x: Input tensor
        fallback_kernel: Original PyTorch implementation

    Returns:
        SiLU(x) = x * sigmoid(x)
    """
    # Check conditions for using Triton implementation
    use_triton = (
        x.dtype == torch.bfloat16 and
        x.numel() >= 16 * 1024 * 1024 and
        x.is_cuda
    )

    if use_triton:
        # Lazily import the kernels (and triton) on first call
        from .triton_kernels import triton_silu_kernel_launcher
        try:
            return triton_silu_kernel_launcher(x)
        except Exception:
            # If Triton fails for any reason, fall back to PyTorch
            pass

    # Fall back to PyTorch implementation
    return fallback_kernel.call_boxed(dispatch_keys, x)


def triton_silu_inplace_dispatch(
    dispatch_keys: torch.DispatchKeySet,
    x: torch.Tensor,
    *,
    fallback_kernel,
) -> torch.Tensor:
    """
    Conditional dispatch for Triton SiLU inplace implementation.

    Uses Triton implementation when:
    - Input dtype is bfloat16 AND
    - Input has at least 16M elements

    Otherwise falls back to PyTorch's native inplace implementation.

    Args:
        dispatch_keys: PyTorch dispatch keys
        x: Input tensor to be modified in-place
        fallback_kernel: Original PyTorch inplace implementation

    Returns:
        The input tensor x, modified in-place with SiLU(x) = x * sigmoid(x)
    """
    # Check conditions for using Triton implementation (same as out-of-place)
    use_triton = (
        x.dtype == torch.bfloat16 and
        x.numel() >= 16 * 1024 * 1024 and
        x.is_cuda
    )

    if use_triton:
        # Lazily import the kernels (and triton) on first call
        from .triton_kernels import triton_silu_inplace_kernel_launcher
        try:
            return triton_silu_inplace_kernel_launcher(x)
        except Exception:
            # If Triton fails for any reason, fall back to PyTorch
            pass

    # Fall back to PyTorch inplace implementation
    return fallback_kernel.call_boxed(dispatch_keys, x)


def register_to_dispatcher():
    """
    Register the conditional Triton SiLU overrides to the dispatcher.
    Registers both out-of-place (aten::silu) and inplace (aten::silu_) versions.
    """
    import functools

    # Register out-of-place version (aten::silu)
    fallback_kernel = torch.library.get_kernel("aten::silu", "CUDA")
    dispatch_fn = functools.partial(
        triton_silu_dispatch,
        fallback_kernel=fallback_kernel
    )
    tu.register_op_override(
        "aten",
        "silu",
        "CUDA",
        dispatch_fn,
        allow_multiple_override=False,
        unconditional_override=False,  # We want fallback capability
    )

    # Register inplace version (aten::silu_)
    fallback_inplace_kernel = torch.library.get_kernel("aten::silu_", "CUDA")
    inplace_dispatch_fn = functools.partial(
        triton_silu_inplace_dispatch,
        fallback_kernel=fallback_inplace_kernel
    )
    tu.register_op_override(
        "aten",
        "silu_",
        "CUDA",
        inplace_dispatch_fn,
        allow_multiple_override=False,
        unconditional_override=False,  # We want fallback capability
    )


# Register when this module is imported
register_to_dispatcher()