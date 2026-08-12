"""CUDA aten reduction overrides backed by the CuteDSL kernels.

Conditions check capability and fall through to ATen for unsupported calls.
Supported layouts use a fast path or the general TensorIterator decode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ... import cutedsl_utils as cu
from ...utils import capability as cap
from ...utils.lazy import LazyModule


if TYPE_CHECKING:
    from . import kernel_general as kg, traits as T
else:
    # T and kg import `cutlass`, which `import torch` must not do (see
    # test_no_dsl_imports_after_import_torch). Only the *_impl functions touch them.
    T = LazyModule("torch._native.ops.reductions.traits")
    kg = LazyModule("torch._native.ops.reductions.kernel_general")


# Compute-capability majors this family's kernels have been run on: Hopper and Blackwell.
_ARCH_MAJORS = (9, 10)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Dtypes the final store can produce, so no trailing cast launch (see _run1). Bool is
# absent: cute's Boolean is 1-BIT and the SIMT copy atom rejects it.
_STORABLE_DTYPES = _SUPPORTED_DTYPES


def _acc_policy():
    # Lazy: the values are `cutlass` dtypes, so resolving at import would pull cutlass in.
    # INPUT torch dtype -> (accumulator cute dtype, that accumulator's torch dtype).
    import cutlass

    return {
        torch.float16: (cutlass.Float32, torch.float32),
        torch.bfloat16: (cutlass.Float32, torch.float32),
        torch.float32: (cutlass.Float32, torch.float32),
    }


def _normalize_dims(dim, ndim: int):
    """Normalize aten dimensions, returning None for reduce-all."""
    if dim is None:
        return None
    dims = [dim] if isinstance(dim, int) else list(dim)
    if len(dims) == 0:
        return None
    return {d % ndim for d in dims}


def _dims_ok(dim, ndim: int) -> bool:
    # Decline any dim aten would reject, WITHOUT raising -- a cond must never throw. At
    # ndim 0 only None / [] / 0 / -1 are valid, and `d % ndim` is undefined there.
    if ndim == 0:
        return False
    if ndim > 64:
        return False
    if dim is None:
        return True
    dims = [dim] if isinstance(dim, int) else list(dim)
    if len(dims) == 0:
        return True
    for d in dims:
        if d < -ndim or d >= ndim:  # out of range -> let aten raise
            return False
    norm = [d % ndim for d in dims]
    if len(set(norm)) != len(norm):  # duplicate dims -> let aten raise
        return False
    return True


def _geometry_supported(x: torch.Tensor, dim, nouts=1, has_index=False) -> bool:
    # CAPABILITY only -- no layout declines, since the general arm addresses via the TI
    # decode. What does: an empty axis with no identity, and an index reduce-all of a
    # non-contiguous input, where ATen's flat index is not the storage offset.
    if x.numel() == 0:
        return False
    red = _normalize_dims(dim, x.dim())
    return not (has_index and red is None and not x.is_contiguous())


def _keepdim_reshape(out: torch.Tensor, x_shape, red: set | None, keepdim: bool):
    # Restore the size-1 reduced dims. resize_ not reshape: an aten reduction NEVER aliases
    # its output and view-ness is observable (see kernel_general._as_shape).
    if not keepdim:
        return out
    if red is None:
        target = [1] * len(x_shape)
    else:
        target = [1 if i in red else s for i, s in enumerate(x_shape)]
    return out.reshape(target)


# ----------------------------------------------------------- shared cond helpers


def _out_dtype(self: torch.Tensor, dtype) -> torch.dtype:
    # aten's rule for a value reduction: the explicit `dtype` if given, else the input's own.
    return dtype if dtype is not None else self.dtype


def _supported_out_dtype(dtype) -> bool:
    # We can deliver any of our supported float out dtypes (we accumulate in fp32
    # and cast down). A non-float requested dtype falls back to aten.
    return dtype is None or dtype in _SUPPORTED_DTYPES


def _base_cond(self, dim, nouts=1, has_index=False) -> bool:
    # Shared capability gate; never raises, since a throwing cond crashes the dispatcher
    # instead of falling back. NEG/CONJ declines: the exported buffer holds the UNNEGATED
    # values, and resolving the bit here would re-enter our own copy_ override. An unaligned
    # base declines because the compiled plan bakes the load width its wrap claimed.
    return (
        not cap.is_traced(self)
        and cap.device_ok(self, _ARCH_MAJORS)
        and self.dtype in _SUPPORTED_DTYPES
        and cap.on_current_device(self)
        and not self.is_neg()
        and not self.is_conj()
        and self.const_data_ptr() % 16 == 0
        and _dims_ok(dim, self.dim())
        and _geometry_supported(self, dim, nouts, has_index)
    )


def _run1(make_trait, key, self, red, keepdim, out_torch_dtype):
    # Every single-output reduction override funnels through here: the accumulator comes from
    # _acc_policy, and the result is aten's output dtype for this op.
    acc, kout = _acc_policy()[self.dtype]
    trait = make_trait(acc)
    # STORE aten's output dtype where the store can produce it: casting the fp32 accumulator
    # afterwards is a whole extra ~2us launch per call. Accumulation is untouched.
    kern_out = out_torch_dtype if out_torch_dtype in _STORABLE_DTYPES else kout
    if red is None:
        out = kg.reduce_all(trait, key, self, kern_out)
    else:
        out = kg.reduce_dim(trait, key, self, sorted(red), kern_out)
    out = _keepdim_reshape(out, self.shape, red, keepdim)
    return out.to(out_torch_dtype)


# --- sum / mean: fp32-accumulated, optional out dtype= (else input dtype). ---


def _sum_cond(self, dim=None, keepdim=False, *, dtype=None):
    # No yield to the inner-tree override, deliberately: it registers FIRST and the router is
    # first-match-wins, so yielding would hand what IT declines to aten rather than serving it.
    return _base_cond(self, dim) and _supported_out_dtype(dtype)


def _sum_impl(self, dim=None, keepdim=False, *, dtype=None):
    red = _normalize_dims(dim, self.dim())
    odt = _out_dtype(self, dtype)
    return _run1(lambda acc: T.SumOps(acc=acc), "sum", self, red, keepdim, odt)


def _mean_cond(self, dim=None, keepdim=False, *, dtype=None):
    return _base_cond(self, dim) and _supported_out_dtype(dtype)


def _mean_impl(self, dim=None, keepdim=False, *, dtype=None):
    red = _normalize_dims(dim, self.dim())
    odt = _out_dtype(self, dtype)
    return _run1(lambda acc: T.MeanOps(acc=acc), "mean", self, red, keepdim, odt)


# --- amax / amin / prod: single-output VALUE reductions; output dtype follows the
# input (amax/amin) or the optional out dtype= (prod). ---


def _amax_impl(self, dim=(), keepdim=False):
    red = _normalize_dims(dim, self.dim())
    return _run1(lambda acc: T.AMaxOps(acc=acc), "amax", self, red, keepdim, self.dtype)


def _amin_impl(self, dim=(), keepdim=False):
    red = _normalize_dims(dim, self.dim())
    return _run1(lambda acc: T.AMinOps(acc=acc), "amin", self, red, keepdim, self.dtype)


def _prod_impl(self, dim, keepdim=False, *, dtype=None):
    red = _normalize_dims(dim, self.dim())
    odt = _out_dtype(self, dtype)
    return _run1(lambda acc: T.ProdOps(acc=acc), "prod", self, red, keepdim, odt)


def _amax_cond(self, dim=(), keepdim=False):
    return _base_cond(self, dim)


def _amin_cond(self, dim=(), keepdim=False):
    return _base_cond(self, dim)


def _prod_cond(self, dim, keepdim=False, *, dtype=None):
    # First-match-wins leaves the inner-tree prod override its eligible calls, as in
    # _sum_cond; we take what it declines rather than handing those to aten.
    return _base_cond(self, dim) and _supported_out_dtype(dtype)


# --- Group B: INDEX reductions. The traits carry the index in a second accumulator field,
# so these route through the index-aware paths. Ties and NaNs match aten (first wins). ---


def _idx_width(self, red):
    # The in-kernel INDEX accumulator: the winning position ranges over the REDUCED extent, so
    # Int32 overflows at 2**31. The OUTPUT buffer is always int64, aten's index dtype.
    import cutlass

    if red is None:
        extent = self.numel()
    else:
        extent = 1
        for d in red:
            extent *= self.shape[d]
    if extent > (1 << 31) - 1:
        return cutlass.Int64, "i64"
    return cutlass.Int32, "i32"


def _run_arg(make_trait, key, self, red, keepdim):
    # The kernel STORES int64 directly, since a trailing widening cast is a whole extra ~2us
    # launch; the accumulator keeps _idx_width's narrower type so shuffles stay cheap.
    acc, _ = _acc_policy()[self.dtype]
    idx_cute, tag = _idx_width(self, red)
    trait = make_trait(acc, idx_cute)
    key = key + tag  # distinct idx width -> distinct compiled kernel
    if red is None:
        out = kg.reduce_all(trait, key, self, torch.int64)
    else:
        out = kg.reduce_dim(trait, key, self, sorted(red), torch.int64)
    return _keepdim_reshape(out, self.shape, red, keepdim)


def _run_dim2(make_trait, key, self, dim, keepdim):
    # Values keep the input dtype; the index is stored as int64 directly, as in _run_arg. Via
    # _normalize_dims, not `dim % self.dim()`, which is undefined for a 0-dim input.
    acc, _ = _acc_policy()[self.dtype]
    red = {dim % self.dim()}
    idx_cute, tag = _idx_width(self, red)
    trait = make_trait(acc, idx_cute)
    dts = [self.dtype, torch.int64]
    vals, idxs = kg.reduce_dim2(trait, key + tag, self, sorted(red), dts)
    vals = _keepdim_reshape(vals, self.shape, red, keepdim)
    idxs = _keepdim_reshape(idxs, self.shape, red, keepdim)
    return vals, idxs


def _argmax_impl(self, dim=None, keepdim=False):
    red = _normalize_dims(dim, self.dim())
    return _run_arg(
        lambda acc, idx: T.ArgMaxOps(acc=acc, idx=idx), "argmax", self, red, keepdim
    )


def _argmin_impl(self, dim=None, keepdim=False):
    red = _normalize_dims(dim, self.dim())
    return _run_arg(
        lambda acc, idx: T.ArgMinOps(acc=acc, idx=idx), "argmin", self, red, keepdim
    )


def _max_dim_impl(self, dim, keepdim=False):
    return _run_dim2(
        lambda acc, idx: T.MaxDimOps(acc=acc, idx=idx), "max.dim", self, dim, keepdim
    )


def _min_dim_impl(self, dim, keepdim=False):
    return _run_dim2(
        lambda acc, idx: T.MinDimOps(acc=acc, idx=idx), "min.dim", self, dim, keepdim
    )


def _argmax_cond(self, dim=None, keepdim=False):
    return _base_cond(self, dim, has_index=True)


def _argmin_cond(self, dim=None, keepdim=False):
    return _base_cond(self, dim, has_index=True)


def _max_dim_cond(self, dim, keepdim=False):
    # max.dim/min.dim take a required single int dim (not a list); _base_cond's
    # _dims_ok accepts the int form and declines scalars / out-of-range.
    return _base_cond(self, dim, nouts=2, has_index=True)


def _min_dim_cond(self, dim, keepdim=False):
    return _base_cond(self, dim, nouts=2, has_index=True)


def register_reduction_overrides() -> None:
    # CUDA overrides; cu.register_op_override short-circuits when the CuteDSL
    # runtime is unavailable, so this is safe to call unconditionally at import.
    cu.register_op_override(
        "aten", "sum.dim_IntList", "CUDA", cond=_sum_cond, impl=_sum_impl
    )
    cu.register_op_override(
        "aten", "mean.dim", "CUDA", cond=_mean_cond, impl=_mean_impl
    )
    # Group A: amax / amin / prod (single-output value reductions).
    cu.register_op_override("aten", "amax", "CUDA", cond=_amax_cond, impl=_amax_impl)
    cu.register_op_override("aten", "amin", "CUDA", cond=_amin_cond, impl=_amin_impl)
    cu.register_op_override(
        "aten", "prod.dim_int", "CUDA", cond=_prod_cond, impl=_prod_impl
    )
    # Group B: argmax / argmin (int64 index) and max.dim / min.dim (values, indices).
    cu.register_op_override(
        "aten", "argmax", "CUDA", cond=_argmax_cond, impl=_argmax_impl
    )
    cu.register_op_override(
        "aten", "argmin", "CUDA", cond=_argmin_cond, impl=_argmin_impl
    )
    cu.register_op_override(
        "aten", "max.dim", "CUDA", cond=_max_dim_cond, impl=_max_dim_impl
    )
    cu.register_op_override(
        "aten", "min.dim", "CUDA", cond=_min_dim_cond, impl=_min_dim_impl
    )
