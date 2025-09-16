"""Functional interface for scaled_mm and friends"""

from enum import Enum
from typing import Callable, Optional, TYPE_CHECKING, Union, List

import torch
from torch import _VF, sym_int as _sym_int, Tensor

class ScalingType(Enum):
    Tensorwise = 0
    Rowwise = 1
    Blockwise_1x16 = 2
    Blockwise_1x32 = 3
    Blockwise_1x128 = 4
    Blockwise_128x128 = 5

class SwizzleType(Enum):
    # No swizzling
    NoSwizzle = 0
    # NVIDIA Blockwell-style swizzle
    Swizzle_32_4_4 = 1


def scaled_mm(
    mat_a: Tensor,
    mat_b: Tensor,
    scale_a: List[Tensor],
    scale_recipe_a: List[ScalingType],
    scale_b: List[Tensor],
    scale_recipe_b: List[ScalingType],
    swizzle_a: List[SwizzleType] = None,
    swizzle_b: List[SwizzleType] = None,
    bias: Tensor = None,
    output_dtype: torch.dtype = torch.bfloat16,
    scale_output: List[Tensor] = None,
    scale_recipe_output: List[ScalingType] = None,
    contraction_dim: List[int] = (-1, -2)):

    def expand_single_value(v):
        if not isinstance(v, (list, tuple)) and v is not None:
            return [v, ]
        else:
            return v

    scale_a = expand_single_value(scale_a)
    scale_recipe_a = expand_single_value(scale_recipe_a)
    scale_b = expand_single_value(scale_b)
    scale_recipe_b = expand_single_value(scale_recipe_b)
    # native_functions has restrictions on what can be defined
    # & passed through - std::optional<ArrayRef<Tensor>> for instance
    # *cannot* be passed, but an empty vector (list) can.
    # So, we need to convert None arguments for lists in python
    # explicitly into empty lists.
    def list_or_empty(l: List):
        return [] if not l else l

    def enum_list_as_int_list(l: List):
        return [li.value for li in l]

    print(f'{scale_recipe_a=}')

    out = torch._scaled_mm_v2(
            mat_a,
            mat_b,
            scale_a,
            enum_list_as_int_list(scale_recipe_a),
            enum_list_as_int_list(list_or_empty(swizzle_a)),
            scale_b,
            enum_list_as_int_list(scale_recipe_b),
            enum_list_as_int_list(list_or_empty(swizzle_b)),
            bias,
            output_dtype,
            list_or_empty(scale_output),
            enum_list_as_int_list(list_or_empty(scale_recipe_output)),
            contraction_dim)

    # If output is high precision, we'll get a single return.
    # Otherwise, will get [output, scale0, ... scaleN] for N-level scaling
    #  - split the latter case into output, [scale0, ..., scaleN]
    if len(out) == 1:
        return out
    else:
        return out[:1], out[1:]
