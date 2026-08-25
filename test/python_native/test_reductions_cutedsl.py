# Owner(s): ["module: dsl-native-ops"]
#
# Wiring tests for routing, fallback, and CUDA graph capture. OpInfo tests cover
# numerical behavior.

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


def _disabled():
    return torch.backends.python_native.cutedsl.disabled()


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestCuTeDSLReductionWiring(TestCase):
    def _fired_count(self, fn):
        from torch._native.ops.reductions import kernel_general as kg

        names = ("reduce_dim", "reduce_dim2", "reduce_all")
        orig = {nm: getattr(kg, nm) for nm in names}
        n = [0]

        def wrap(f):
            def counting(*a, **k):
                n[0] += 1
                return f(*a, **k)

            return counting

        for nm in names:
            setattr(kg, nm, wrap(orig[nm]))
        try:
            fn()
        finally:
            for nm in names:
                setattr(kg, nm, orig[nm])
        return n[0]

    def test_supported_call_fires(self):
        # A supported call (CUDA, float, contiguous, valid dim) must route through
        # our kernel -- guards against a silently-all-fallback regression.
        x = torch.randn(128, 512, device="cuda")
        self.assertEqual(self._fired_count(lambda: torch.sum(x, dim=-1)), 1)
        self.assertEqual(self._fired_count(lambda: torch.mean(x, dim=-1)), 1)
        self.assertEqual(self._fired_count(lambda: torch.amax(x, dim=-1)), 1)
        # Group B: single-output index (argmax) and two-output (max.dim).
        self.assertEqual(self._fired_count(lambda: torch.argmax(x, dim=-1)), 1)
        self.assertEqual(self._fired_count(lambda: torch.max(x, dim=-1)), 1)
        # Group C: parameterized / non-float-output single reductions.
        self.assertEqual(self._fired_count(lambda: torch.var(x, dim=-1)), 1)
        self.assertEqual(
            self._fired_count(lambda: torch.linalg.vector_norm(x, dim=-1)), 1
        )
        self.assertEqual(self._fired_count(lambda: torch.count_nonzero(x, dim=-1)), 1)
        # Group D: two float-output reductions.
        self.assertEqual(self._fired_count(lambda: torch.var_mean(x, dim=-1)), 1)
        self.assertEqual(self._fired_count(lambda: torch.aminmax(x, dim=-1)), 1)

    def test_unsupported_dtype_falls_back(self):
        # Integer input is outside the supported set -> must NOT hit our kernel.
        xi = torch.randint(0, 9, (64, 64), device="cuda")
        self.assertEqual(self._fired_count(lambda: torch.sum(xi, dim=-1)), 0)
        # ... and the result is still correct (served by aten).
        with _disabled():
            ref = torch.sum(xi, dim=-1)
        self.assertEqual(torch.sum(xi, dim=-1), ref)

    def test_noncontiguous_is_served(self):
        # A layout is never a reason to decline: the general arm addresses through the TI
        # offset decode, so a non-contiguous input is SERVED, not handed back.
        xt = torch.randn(64, 128, device="cuda").t()
        for fn in (
            lambda t: torch.sum(t, dim=-1),
            lambda t: torch.sum(t, dim=0),
            lambda t: torch.sum(t),  # reduce-ALL of a non-contiguous input
            lambda t: torch.amax(t, dim=-1),
            lambda t: torch.var(t, dim=-1),
            lambda t: torch.argmax(t, dim=-1),
        ):
            with self.subTest(fn=fn):
                self.assertEqual(self._fired_count(lambda: fn(xt)), 1)
                with _disabled():
                    ref = fn(xt)
                self.assertEqual(fn(xt), ref)

    def test_empty_reduction_is_served_or_declined_by_identity(self):
        # An empty reduction is an empty tensor (a KEPT extent is zero) or the op's IDENTITY over the
        # kept shape. The max/min family has none and aten RAISES, so serving it would answer where
        # aten errors.
        empty_out = torch.randn(0, 5, device="cuda")
        empty_axis = torch.randn(5, 0, device="cuda")
        for fn in (
            lambda t: t.sum(dim=1),
            lambda t: t.mean(dim=1),
            lambda t: torch.prod(t, 1),
            lambda t: t.all(dim=1),
            lambda t: torch.count_nonzero(t, dim=1),
            lambda t: torch.linalg.vector_norm(t, dim=1),
        ):
            for x in (empty_out, empty_axis):
                with self.subTest(fn=fn, shape=tuple(x.shape)):
                    with _disabled():
                        ref = fn(x)
                    got = fn(x)
                    self.assertEqual(got.shape, ref.shape)
                    self.assertEqual(got.dtype, ref.dtype)
                    self.assertEqual(got, ref, exact_dtype=True)
        # No identity + a non-empty output -> aten must raise, so we must not answer. aten
        # reports this as an IndexError (TORCH_CHECK_INDEX on the reduced dim).
        for fn in (torch.amax, torch.amin, torch.argmax, torch.aminmax):
            with self.subTest(fn=fn):
                with self.assertRaises((RuntimeError, IndexError)):
                    fn(empty_axis, dim=1)
        # ... but the same ops over an EMPTY output are served, since no identity is needed.
        for fn in (torch.amax, torch.amin, torch.argmax):
            with self.subTest(fn=fn, empty_out=True):
                self.assertEqual(fn(empty_out, dim=1).shape, (0,))

    @staticmethod
    def _compiled_kernel_count():
        # Every compiled reduction kernel lands in exactly one of these caches, keyed on its
        # compile signature -- so len() IS the number of distinct kernels built so far.
        from torch._native.ops.reductions import (
            kernel_coltile,
            kernel_general,
            kernel_rowtile,
            kernel_xcta,
        )

        caches = (
            kernel_general._COMPILE_CACHE,
            kernel_rowtile._CACHE,
            kernel_coltile._CACHE,
            kernel_xcta._PLAN,
        )
        return sum(len(c) for c in caches), caches

    def test_kernel_count_does_not_scale_with_shape(self):
        # GUARD. The design rests on compiling O(op x dtype x structure) kernels, not O(shapes): a
        # size-derived const_expr breaks that and shows up as compile time, not a wrong answer. The
        # bound is RELATIVE so it survives adding kernels. These N share a vec class and one bucket
        # rung, so a correct stack compiles the same kernels for two of them as for ten. An order
        # that fixes its add DAG at compile time cannot satisfy this and is opt-in, so off here.
        few = [4096, 4104]
        many = [4096, 4104, 4112, 4120, 4128, 4136, 4144, 4152, 4160, 4168]

        def run(sizes):
            _, caches = self._compiled_kernel_count()
            for c in caches:
                c.clear()
            for n in sizes:
                x = torch.randn(512, n, device="cuda")
                torch.sum(x, dim=-1)
                torch.amax(x, dim=-1)
                # column path too: its split factor and stage-2 mapping are shape-derived
                torch.sum(x, dim=0)
                del x
            return self._compiled_kernel_count()[0]

        n_few = run(few)
        n_many = run(many)
        self.assertEqual(
            n_many,
            n_few,
            f"compiled kernels grew from {n_few} (2 shapes) to {n_many} (10 shapes): a "
            f"size-derived const_expr has crept back in",
        )

    def test_scalar_is_served(self):
        # A 0-dim input reduces to ITSELF and aten accepts exactly dim None/[]/0/-1, so the result is
        # 0-dim whatever keepdim says. The normalization must not compute `d % ndim` with ndim 0.
        s = torch.tensor(3.5, device="cuda")
        for fn in (
            lambda t: torch.sum(t),
            lambda t: torch.sum(t, dim=0),
            lambda t: torch.sum(t, dim=-1),
            lambda t: torch.sum(t, dim=0, keepdim=True),
            lambda t: torch.amax(t),
            lambda t: torch.argmax(t),
            lambda t: torch.max(t, dim=0),
            lambda t: torch.aminmax(t, dim=0),
            lambda t: torch.count_nonzero(t),
        ):
            with self.subTest(fn=fn):
                self.assertGreaterEqual(self._fired_count(lambda: fn(s)), 1)
                with _disabled():
                    ref = fn(s)
                self.assertEqual(fn(s), ref, exact_dtype=True)
        # An out-of-range or duplicated dim still has to reach aten's error.
        for bad in (1, -2, [0, -1]):
            with self.subTest(dim=bad), self.assertRaises((RuntimeError, IndexError)):
                torch.sum(s, dim=bad)

    def test_invalid_dim_defers_to_aten(self):
        # dim args aten rejects (out-of-range / duplicate) must surface aten's
        # normal error -- the cond declines so aten validates, no wrapped result.
        x = torch.randn(4, 5, 6, device="cuda")
        with self.assertRaises(IndexError):
            torch.sum(x, dim=3)
        with self.assertRaises(RuntimeError):
            torch.sum(x, dim=(0, 0))

    def test_cow_input_served_and_preserved(self):
        # A COW input is SERVED (it exports read-only, so from_dlpack reads const_data_ptr()) and
        # must STAY COW: reading it must not materialize it.
        base = torch.randn(128, 512, device="cuda")
        x = torch._lazy_clone(base)
        self.assertEqual(self._fired_count(lambda: torch.sum(x, dim=-1)), 1)
        self.assertTrue(torch._C._is_cow_tensor(x))

    def test_fast_geometry_routing(self):
        # `fast_kind` is a ROUTER, not a gate: a geometry it cannot reshape onto the fast paths is
        # served by the general arm. So every case here fires; only which path takes it differs.
        served = [
            ("2D last-dim", torch.randn(512, 512, device="cuda"), -1),
            ("2D dim0", torch.randn(512, 512, device="cuda"), 0),
            (
                "3D last-dim coalesces to row",
                torch.randn(64, 32, 512, device="cuda"),
                -1,
            ),
            (
                "3D dims (1,2) coalesce to row",
                torch.randn(128, 32, 32, device="cuda"),
                (1, 2),
            ),
        ]
        for name, x, dim in served:
            self.assertEqual(
                self._fired_count(lambda: torch.sum(x, dim=dim)),
                1,
                f"{name} should fire",
            )
        # These reach no fast path (mid-dim, transposed, gapped, window-overlapping), so the
        # general arm serves them -- and must get the same answer aten does.
        general = [
            ("3D mid-dim", torch.randn(512, 512, 64, device="cuda"), 1),
            ("transposed", torch.randn(512, 512, device="cuda").t(), -1),
            ("gapped slice", torch.randn(512, 512, device="cuda")[:, ::2], -1),
            (
                "permuted 3D",
                torch.randn(32, 64, 128, device="cuda").permute(2, 0, 1),
                2,
            ),
            (
                "overlapping windows",
                torch.randn(64, 128, device="cuda").unfold(1, 4, 2),
                -1,
            ),
        ]
        for name, x, dim in general:
            self.assertEqual(
                self._fired_count(lambda: torch.sum(x, dim=dim)),
                1,
                f"{name} must be served, not declined",
            )
            with _disabled():
                ref = torch.sum(x, dim=dim)
            self.assertEqual(torch.sum(x, dim=dim), ref, atol=1e-3, rtol=1e-3)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "needs >= 2 GPUs")
    def test_other_device_defers(self):
        # A tensor not on the current device must fall back (kernel/stream caches
        # are current-device-bound). cuda:1 with cuda:0 current -> aten.
        x = torch.randn(128, 512, device="cuda:1")
        self.assertEqual(self._fired_count(lambda: torch.sum(x, dim=-1)), 0)

    def test_graph_capturable(self):
        # The override must capture into a CUDA graph and replay correctly (the
        # earlier _stream() bug made cute launches deadlock / produce empty graphs).
        x = torch.randn(8192, 1024, device="cuda")
        f = lambda: torch.sum(x, dim=-1)  # noqa: E731
        with _disabled():
            ref = f()
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                f()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = f()
        g.replay()
        torch.cuda.synchronize()
        self.assertEqual(out, ref, atol=1e-2, rtol=1e-2)

    def test_sub_warp_row_width_does_not_crash(self):
        # REGRESSION: the small-N rungs returned 8/16 threads per row and warps_per_row floored to 0,
        # a trace-time ZeroDivisionError on an ordinary sum. Non-monotonic in N, so cover the range.
        for n in (8, 16, 24, 32, 33, 48, 63, 64, 96, 128, 192):
            x = torch.rand(257, n, device="cuda")
            self.assertEqual(
                torch.sum(x, dim=1), x.double().sum(dim=1).float(), atol=1e-3, rtol=1e-3
            )
            torch.linalg.vector_norm(x, 2, dim=1)  # same reduce path, must not raise

    def test_strided_single_element_view_is_served(self):
        # REGRESSION: is_contiguous() is True for ANY single-element tensor, since with one element
        # the stride is unobservable -- so a.diagonal(offset=2) is a contiguous shape-(1,) tensor that
        # still declares stride (4,), which the DSL rejected outright. The wrap restrides it, and
        # these are SERVED: declining would give up coverage for a difference nothing can observe.
        a = torch.randn(5, 3, device="cuda", dtype=torch.float64)
        d = a.diagonal(offset=2)
        self.assertEqual(d.shape, torch.Size([1]))
        self.assertNotEqual(d.stride(), (1,))  # the leftover stride is the whole point
        for fn in (lambda z: z.sum(), lambda z: z.mean()):
            with _disabled():
                ref = fn(d)
            self.assertEqual(fn(d), ref)

    def test_misaligned_and_lazy_metadata_inputs_decline(self):
        # Two cond gates that are not numerics: an unaligned base pointer, because the compiled plan
        # bakes the load width its wrap claimed and cannot serve both; and a NEG/CONJ bit, which is
        # lazy metadata, so the exported buffer holds the unnegated values and resolving it here would
        # re-enter our own copy_ override.
        base = torch.arange(4096, device="cuda", dtype=torch.float64) + 1
        for off in (0, 1, 2, 3):
            t = base[off : off + 512].view(256, 2)
            for fn in (
                lambda z: z.sum(dim=0),
                lambda z: z.sum(dim=1),
                lambda z: z.sum(),
            ):
                with _disabled():
                    ref = fn(t)
                self.assertEqual(fn(t), ref, f"off={off}")
        n = torch.randn(64, device="cuda", dtype=torch.float64)._neg_view()
        self.assertTrue(n.is_neg())
        for fn in (lambda z: z.sum(dim=0), lambda z: z.sum(), lambda z: z.mean()):
            with _disabled():
                ref = fn(n)
            self.assertEqual(fn(n), ref)


if __name__ == "__main__":
    run_tests()
