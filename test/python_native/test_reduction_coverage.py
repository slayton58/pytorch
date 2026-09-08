# Owner(s): ["module: dsl-native-ops"]
#
# COVERAGE as a matrix rather than a sample: for every (op, dtype, options) cell in the
# declared set, does OUR kernel serve the call? A SILENT DECLINE is what this catches and no
# other test can -- ATen still answers correctly, so a value-only test sees nothing.
#
# Served-ness is read at the capability COND, not at a launch: the empty and 0-dim shapes are
# answered without launching anything, which a kernel-level probe cannot tell from a decline.

import sys
import unittest
from collections import Counter

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TEST_CUTEDSL, TestCase


if not TEST_CUTEDSL:
    sys.stderr.write("CuTeDSL not available\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("CuTeDSL not available")

from torch._native import registry as R


_OVERRIDES_MODULE = "torch._native.ops.reductions.overrides"

# The dtypes the overrides DECLARE (see overrides._ENVELOPE). Widening that envelope means adding a
# row here in the same commit, which is what makes this a coverage claim rather than a smoke test.
DTYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
}

OPS = {
    "sum": lambda t, **k: torch.sum(t, **k),
    "mean": lambda t, **k: torch.mean(t, **k),
    "nansum": lambda t, **k: torch.nansum(t, **k),
    "prod": lambda t, **k: (
        torch.prod(t, k["dim"], k.get("keepdim", False))
        if "dim" in k
        else torch.prod(t)
    ),
    "amax": lambda t, **k: torch.amax(t, **k),
    "amin": lambda t, **k: torch.amin(t, **k),
    "argmax": lambda t, **k: torch.argmax(t, **k),
    "max.dim": lambda t, **k: torch.max(t, **k) if "dim" in k else torch.max(t),
    "min.dim": lambda t, **k: torch.min(t, **k) if "dim" in k else torch.min(t),
    "aminmax": lambda t, **k: torch.aminmax(t, **k),
    "var": lambda t, **k: torch.var(t, **k),
    "std": lambda t, **k: torch.std(t, **k),
    "var_mean": lambda t, **k: torch.var_mean(t, **k),
    "std_mean": lambda t, **k: torch.std_mean(t, **k),
    "vector_norm": lambda t, **k: torch.linalg.vector_norm(t, **k),
    "all": lambda t, **k: torch.all(t, **k),
    "any": lambda t, **k: torch.any(t, **k),
    "count_nonzero": lambda t, **k: torch.count_nonzero(t, **k),
}

# The OPTIONS axis: dim spelling (single / negative / tuple / absent), keepdim, and the geometries
# that used to be declined -- a transposed input, an empty extent, a 0-dim input.
OPTIONS = [
    ("dim=-1", {"dim": -1}, {}),
    ("dim=0", {"dim": 0}, {}),
    ("dim=(0,1)", {"dim": (0, 1)}, {}),
    ("keepdim", {"dim": -1, "keepdim": True}, {}),
    ("full", {}, {}),
    ("noncontig", {"dim": -1}, {"contiguous": False}),
    ("empty-axis", {"dim": 1}, {"empty": True}),
    ("0-dim", {"dim": 0}, {"zerodim": True}),
]

EXPECTED_DECLINES = set()


def _build(dtype, contiguous=True, empty=False, zerodim=False):
    if zerodim:
        return torch.zeros((), device="cuda").to(dtype)
    shape = (0, 5) if empty else (64, 128)
    t = torch.randn(shape, device="cuda", dtype=torch.float32).to(dtype)
    return t if contiguous else t.transpose(0, 1)


def _served(fn):
    """Run `fn` and report whether a reduction condition accepted it."""

    def is_ours(node):
        return getattr(node.cond_fn, "__module__", "") == _OVERRIDES_MODULE

    accepted = [False]

    def wrap(f):
        def probe(*args, **kwargs):
            r = f(*args, **kwargs)
            if r:
                accepted[0] = True
            return r

        return probe

    ours = {k: v for k, v in R._graphs.items() if any(is_ours(n) for n in v)}
    saved = [(n, n.cond_fn) for nodes in ours.values() for n in nodes if is_ours(n)]
    for n, _ in saved:
        n.cond_fn = wrap(n.cond_fn)
    for (op_symbol, key), nodes in ours.items():
        R._register_overrides_from_graph(op_symbol, key, nodes)
    try:
        out = fn()
    finally:
        for n, orig in saved:
            n.cond_fn = orig
        for (op_symbol, key), nodes in ours.items():
            R._register_overrides_from_graph(op_symbol, key, nodes)
    return accepted[0], out


@unittest.skipUnless(TEST_CUDA, "CUDA required")
class TestReductionCoverage(TestCase):
    def _agree(self, got, ref):
        pairs = zip(
            got if isinstance(got, tuple) else (got,),
            ref if isinstance(ref, tuple) else (ref,),
        )
        for g, r in pairs:
            if g.dtype != r.dtype:
                return False
            if g.is_floating_point():
                if not torch.allclose(
                    g.float(), r.float(), rtol=1e-2, atol=1e-2, equal_nan=True
                ):
                    return False
            elif not torch.equal(g, r):
                return False
        return True

    def test_every_declared_cell_is_served(self):
        # One pass over (op x dtype x options). Each cell lands in exactly one bucket: ATen rejects
        # it, we serve it and agree, or we decline -- a failure unless it is a documented dof case.
        holes, wrong, counts = [], [], Counter()
        for op_name, fn in OPS.items():
            for dt_name, dtype in DTYPES.items():
                for label, kwargs, build_kw in OPTIONS:
                    cell = (op_name, dt_name, label)
                    t = _build(dtype, **build_kw)
                    call = lambda t=t, fn=fn, kw=kwargs: fn(t, **kw)  # noqa: E731
                    try:
                        with torch.backends.python_native.cutedsl.disabled():
                            ref = call()
                    except Exception:
                        counts["aten-rejects"] += 1
                        continue
                    ok, got = _served(call)
                    if not self._agree(got, ref):
                        wrong.append(cell)
                    elif ok:
                        counts["served"] += 1
                    else:
                        counts["declined"] += 1
                        if cell not in EXPECTED_DECLINES:
                            holes.append(cell)

        self.assertEqual(
            wrong, [], f"cells where our answer differs from ATen: {wrong}"
        )
        self.assertEqual(holes, [], f"cells silently declined to ATen: {holes}")
        # Guard the guard: if the matrix stops exercising the family, the assertions above pass
        # trivially. The floor sits well below the current 716 served, so widening a row is free.
        self.assertGreater(counts["served"], 350)
        self.assertEqual(counts["declined"], len(EXPECTED_DECLINES))

    def test_expected_declines_still_decline(self):
        # The exceptions are a claim about ATen, not a wish: if it stops warning on a 0-dim var this
        # decline is a coverage hole and should be removed rather than kept as a carve-out.
        for op_name, dt_name, _ in sorted(EXPECTED_DECLINES):
            with self.subTest(op=op_name, dtype=dt_name):
                t = _build(DTYPES[dt_name], zerodim=True)
                ok, _ = _served(lambda t=t, op=op_name: OPS[op](t, dim=0))
                self.assertFalse(ok, f"{op_name}/{dt_name} 0-dim is now served")
                with self.assertWarnsRegex(UserWarning, "degrees of freedom"):
                    with torch.backends.python_native.cutedsl.disabled():
                        OPS[op_name](t, dim=0)


if __name__ == "__main__":
    run_tests()
