# The pointwise op definition table (POINTWISE_DEF_TABLE): one PointwiseDef row per
# aten elementwise op. Each row is fully declarative -- the generic registration
# machinery (overrides.py) turns it into a (cond, impl) override, so adding an op is
# one row plus its kernel function in ops.py, not a hand-written override.
#
# This module is deliberately cutlass-FREE: it holds only the metadata registration
# needs (aten name, arity, promotion kind, scalar args, output-dtype policy) so that
# `import torch` -> override registration can read the table without pulling in the
# DSL runtime (the lazy-DSL-import contract; see test_no_dsl_imports_after_import_torch).
# The actual kernel math lives in ops.py, where every `fn` is a @cute.jit function;
# a row references it BY NAME (`fn`, a str) and overrides.py resolves the callable
# lazily via ops.get_fn(name) on the first real (non-declined) call.
#
# The named ops.py function is @cute.jit-able over COMPUTE-dtype scalars:
#   fn(*input_vals, *scalar_consts) -> result | tuple-of-results
# Inputs arrive already converted to the compute dtype; baked scalar args (e.g. add's
# `alpha`) follow, as compute-dtype constants. The result is cast to the op's output
# dtype. `fn` references only DSL ops (operators, cute.math.*), never a user-class
# method (which would trip the IR flattener).

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

import torch
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND as PromotionKind


if TYPE_CHECKING:
    from collections.abc import Callable


class PointwiseDef(NamedTuple):
    aten: str  # aten op symbol incl overload, e.g. "add.Tensor", "neg"
    nin: int  # number of tensor inputs
    fn: str  # name of the @cute.jit kernel function in ops.py (resolved lazily)
    # aten elementwise type-promotion kind (single value, not combinable: a closed
    # Enum keying torch's elementwise_dtypes algorithm, not a bitwise Flag).
    promotion: PromotionKind = PromotionKind.DEFAULT
    scalars: tuple[str, ...] = ()  # positional arg names baked as compute consts
    nout: int = 1  # number of outputs (>1: e.g. frexp)
    # ESCAPE HATCH for ops whose output dtypes are NOT all the promotion result
    # (e.g. frexp -> (float mantissa, int32 exponent)). Maps the promotion result
    # torch dtype -> list[torch dtype] of length nout. None -> every output uses the
    # promotion result dtype (the common case).
    out_dtypes: Callable | None = None
    # Restrict the INPUT dtypes this override serves; inputs outside the set fall
    # back to aten. None -> the family default (all supported floats). Use to narrow
    # an op whose kernel is only correct for some dtypes (e.g. frexp excludes fp64).
    dtypes: tuple | None = None


_DEFAULT = PromotionKind.DEFAULT
_INT2FLOAT = PromotionKind.INT_TO_FLOAT
_BOOL = PromotionKind.ALWAYS_BOOL

POINTWISE_DEF_TABLE: tuple[PointwiseDef, ...] = (
    # --- binary / unary arithmetic (DEFAULT: output dtype = promoted input) ---
    PointwiseDef("neg", 1, "_neg"),
    PointwiseDef("add.Tensor", 2, "_add", scalars=("alpha",)),
    PointwiseDef("sub.Tensor", 2, "_sub", scalars=("alpha",)),
    PointwiseDef("mul.Tensor", 2, "_mul"),
    PointwiseDef("div.Tensor", 2, "_div"),
    PointwiseDef("maximum", 2, "_maximum"),
    PointwiseDef("minimum", 2, "_minimum"),
    PointwiseDef("atan2", 2, "_atan2"),
    # --- rounding / sign / activation (DEFAULT) ---
    PointwiseDef("floor", 1, "_floor"),
    PointwiseDef("ceil", 1, "_ceil"),
    PointwiseDef("trunc", 1, "_trunc"),
    PointwiseDef("sign", 1, "_sign"),
    PointwiseDef("relu", 1, "_relu"),
    # --- unary transcendental math (INT_TO_FLOAT: int input -> float output) ---
    PointwiseDef("exp", 1, "_exp", promotion=_INT2FLOAT),
    PointwiseDef("exp2", 1, "_exp2", promotion=_INT2FLOAT),
    PointwiseDef("expm1", 1, "_expm1", promotion=_INT2FLOAT),
    PointwiseDef("log", 1, "_log", promotion=_INT2FLOAT),
    PointwiseDef("log2", 1, "_log2", promotion=_INT2FLOAT),
    PointwiseDef("log10", 1, "_log10", promotion=_INT2FLOAT),
    PointwiseDef("log1p", 1, "_log1p", promotion=_INT2FLOAT),
    PointwiseDef("sqrt", 1, "_sqrt", promotion=_INT2FLOAT),
    PointwiseDef("rsqrt", 1, "_rsqrt", promotion=_INT2FLOAT),
    PointwiseDef("reciprocal", 1, "_reciprocal", promotion=_INT2FLOAT),
    PointwiseDef("sin", 1, "_sin", promotion=_INT2FLOAT),
    PointwiseDef("cos", 1, "_cos", promotion=_INT2FLOAT),
    PointwiseDef("tan", 1, "_tan", promotion=_INT2FLOAT),
    PointwiseDef("asin", 1, "_asin", promotion=_INT2FLOAT),
    PointwiseDef("acos", 1, "_acos", promotion=_INT2FLOAT),
    PointwiseDef("atan", 1, "_atan", promotion=_INT2FLOAT),
    PointwiseDef("tanh", 1, "_tanh", promotion=_INT2FLOAT),
    PointwiseDef("erf", 1, "_erf", promotion=_INT2FLOAT),
    PointwiseDef("sigmoid", 1, "_sigmoid", promotion=_INT2FLOAT),
    # --- comparisons (ALWAYS_BOOL: output is bool) ---
    PointwiseDef("gt.Tensor", 2, "_gt", promotion=_BOOL),
    PointwiseDef("lt.Tensor", 2, "_lt", promotion=_BOOL),
    PointwiseDef("ge.Tensor", 2, "_ge", promotion=_BOOL),
    PointwiseDef("le.Tensor", 2, "_le", promotion=_BOOL),
    PointwiseDef("eq.Tensor", 2, "_eq", promotion=_BOOL),
    PointwiseDef("ne.Tensor", 2, "_ne", promotion=_BOOL),
    # --- ternary / multi-output ---
    PointwiseDef("addcmul", 3, "_addcmul", scalars=("value",)),
    PointwiseDef(
        "frexp.Tensor",
        1,
        "_frexp",
        nout=2,
        # (mantissa: promotion-result float, exponent: int32) -- the escape hatch.
        out_dtypes=lambda compute: [compute, torch.int32],
        # log2-derived frexp is exact only for fp16/bf16/fp32; fp64 needs bit
        # extraction (deferred). fp64 falls back to aten.
        dtypes=(torch.float16, torch.bfloat16, torch.float32),
    ),
)
