import functools
import logging
import os

from .registry import register_op_registerer, _RegisterFn

log = logging.getLogger(__name__)

CUTEDSL_AVAILABLE = None

log = logging.getLogger(__name__)

def _cutedsl_unavailable_reason() -> None | str:
    deps = [
        ("nvidia-cutlass-dsl", "cutlass"),
        ("apache-tvm-ffi", "tvm_ffi"),
        ("cuda-bindings", "cuda.bindings.driver"),
    ]
    for package_name, module_name in deps:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            return (
                f"missing optional dependency `{package_name}` "
                f"(import `{module_name}` failed: {exc})"
            )
    return None

def _check_cutedsl_runtime_available() -> bool:
    print('running _check_cutedsl_runtime_available() on import')
    global CUTEDSL_AVAILABLE

    if CUTEDSL_AVAILABLE is not None:
        return CUTEDSL_AVAILABLE

    reason = _cutedsl_unavailable_reason()
    if reason is None:
        CUTEDSL_AVAILABLE = True
    else:
        print(
            "scaled_grouped_mm CuTeDSL path requires optional Python packages "
            "`nvidia-cutlass-dsl`, `apache-tvm-ffi`, and `cuda-bindings` "
            "(from NVIDIA cuda-python); "
            f"{reason}"
        )
        CUTEDSL_AVAILABLE = False
    return CUTEDSL_AVAILABLE

_check_cutedsl_runtime_available()

def register_cutedsl_op(fn: _RegisterFn):
    if not CUTEDSL_AVAILABLE:
        return

    register_op_registerer(fn)
